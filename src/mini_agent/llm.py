"""LLM protocol, Chat Completions adapter (DeepSeek/OpenAI), and system prompts."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mini_agent.cost import UsageStats, estimate_tokens_from_text
from mini_agent.messages import Message, Part, TextPart, ToolCallPart, to_chat_messages
from mini_agent.prompt import get_system_prompt as get_system_prompt


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMAuthError(LLMError):
    """Authentication or missing API key error."""


class LLMConnectionError(LLMError):
    """Network connection or timeout error."""


class LLMRateLimitError(LLMError):
    """Rate limit error that may be retried."""


@dataclass
class FunctionCall:
    """Represents a function call requested by the model."""

    name: str
    call_id: str
    arguments: str


@dataclass
class LLMResponse:
    """Unified response extracted from LLM output."""

    text: str | None = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)


class LLMClient(Protocol):
    """Protocol defining the interface between Agent and LLM backend."""

    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResponse:
        """Send conversation messages and tool schemas to LLM and return structured response."""
        ...


def _function_calls_from_accumulator(
    tool_calls_acc: dict[int, dict[str, str]],
) -> list[FunctionCall]:
    """Build FunctionCall list sorted by stream index. Empty ids become invalid_{index}."""
    calls: list[FunctionCall] = []
    for idx in sorted(tool_calls_acc):
        tc_dict = tool_calls_acc[idx]
        call_id = tc_dict["id"] or f"invalid_{idx}"
        calls.append(
            FunctionCall(
                name=tc_dict["name"],
                call_id=call_id,
                arguments=tc_dict["arguments"],
            )
        )
    return calls


def _response_parts(text: str | None, function_calls: list[FunctionCall]) -> list[Part]:
    parts: list[Part] = []
    if text:
        parts.append(TextPart(text=text))
    for call in function_calls:
        parts.append(ToolCallPart(call_id=call.call_id, name=call.name, arguments=call.arguments))
    return parts


def _close_stream(stream: Any) -> None:
    closer = getattr(stream, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


class OpenAIChatCompletionsClient:
    """OpenAI & DeepSeek compatible client with streaming & usage tracking."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        effective_base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        )
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=effective_base_url,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tools schema to standard Chat Completions format."""
        converted = []
        for t in tools:
            if "function" in t:
                converted.append(t)
            else:
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description"),
                            "parameters": t.get("parameters"),
                        },
                    }
                )
        return converted

    def _convert_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize history items to standard Chat Completions messages."""
        messages: list[dict[str, Any]] = []
        for item in history:
            role = item.get("role")
            item_type = item.get("type")

            if item_type == "function_call_output" or role == "tool":
                call_id = item.get("call_id") or item.get("tool_call_id", "")
                output = item.get("output") or item.get("content", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }
                )
            elif role in ("system", "user"):
                messages.append({"role": role, "content": item.get("content", "")})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant"}
                if item.get("content"):
                    msg["content"] = item["content"]
                if item.get("tool_calls"):
                    msg["tool_calls"] = item["tool_calls"]
                messages.append(msg)
            elif item_type == "message":
                messages.append(
                    {
                        "role": item.get("role", "assistant"),
                        "content": item.get("content", ""),
                    }
                )
        return messages

    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResponse:
        """Call Chat Completions API with optional token streaming and usage tracking."""
        import openai

        chat_messages = to_chat_messages(messages)
        chat_tools = self._convert_tools(tools)

        if cancel is not None and cancel.is_set():
            return LLMResponse()

        try:
            if on_token is not None:
                response_stream = self._client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    tools=chat_tools if chat_tools else None,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                accumulated_text = ""
                tool_calls_acc: dict[int, dict[str, str]] = {}
                stream_usage: UsageStats | None = None

                try:
                    for chunk in response_stream:
                        if cancel is not None and cancel.is_set():
                            break
                        if getattr(chunk, "usage", None):
                            u = chunk.usage
                            stream_usage = UsageStats(
                                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                                total_tokens=getattr(u, "total_tokens", 0) or 0,
                            )

                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta.content:
                            accumulated_text += delta.content
                            on_token(delta.content)
                        if getattr(delta, "tool_calls", None):
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc.id:
                                    tool_calls_acc[idx]["id"] += tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_acc[idx]["name"] += tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_acc[idx]["arguments"] += tc.function.arguments
                except KeyboardInterrupt:
                    if cancel is not None:
                        cancel.set()
                finally:
                    _close_stream(response_stream)

                function_calls = _function_calls_from_accumulator(tool_calls_acc)

                if not stream_usage or stream_usage.total_tokens == 0:
                    prompt_str = " ".join([str(m.get("content", "")) for m in chat_messages])
                    p_tok = estimate_tokens_from_text(prompt_str)
                    c_tok = estimate_tokens_from_text(accumulated_text)
                    stream_usage = UsageStats(
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=p_tok + c_tok,
                    )

                return LLMResponse(
                    text=accumulated_text if accumulated_text else None,
                    function_calls=function_calls,
                    parts=_response_parts(
                        accumulated_text if accumulated_text else None, function_calls
                    ),
                    usage=stream_usage,
                )

            # Non-streaming execution
            response = self._client.chat.completions.create(
                model=model,
                messages=chat_messages,
                tools=chat_tools if chat_tools else None,
                stream=False,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthError(f"API 身份验证失败，请检查 API Key 有效性: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMConnectionError(
                f"无法连接到 API 服务，请检查网络或 base_url 配置: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(f"API 速率限制 (Rate Limit): {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"API 调用失败: {exc}") from exc

        if not response.choices:
            return LLMResponse(text=None)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content
        function_calls: list[FunctionCall] = []

        if getattr(msg, "tool_calls", None):
            for i, tc in enumerate(msg.tool_calls):
                function_calls.append(
                    FunctionCall(
                        name=tc.function.name or "",
                        call_id=tc.id or f"invalid_{i}",
                        arguments=tc.function.arguments or "",
                    )
                )

        resp_usage: UsageStats
        if getattr(response, "usage", None):
            u = response.usage
            resp_usage = UsageStats(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )
        else:
            prompt_str = " ".join([str(m.get("content", "")) for m in chat_messages])
            p_tok = estimate_tokens_from_text(prompt_str)
            c_tok = estimate_tokens_from_text(text or "")
            resp_usage = UsageStats(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
            )

        return LLMResponse(
            text=text,
            function_calls=function_calls,
            parts=_response_parts(text, function_calls),
            usage=resp_usage,
        )


# Backward compatibility alias
OpenAIResponsesClient = OpenAIChatCompletionsClient

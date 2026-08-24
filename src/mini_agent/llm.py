"""LLM protocol, Chat Completions adapter (DeepSeek/OpenAI), and system prompts."""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mini_agent.cost import UsageStats, estimate_tokens_from_text
from mini_agent.prompt import get_system_prompt as get_system_prompt


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMAuthError(LLMError):
    """Authentication or missing API key error."""


class LLMConnectionError(LLMError):
    """Network connection or timeout error."""


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
    raw_output: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)


class LLMClient(Protocol):
    """Protocol defining the interface between Agent and LLM backend."""

    def create_response(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send conversation history and tool schemas to LLM and return structured response."""
        ...


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
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Call Chat Completions API with optional token streaming and usage tracking."""
        import openai

        messages = self._convert_messages(history)
        chat_tools = self._convert_tools(tools)

        try:
            if on_token is not None:
                response_stream = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=chat_tools if chat_tools else None,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                accumulated_text = ""
                tool_calls_acc: dict[int, dict[str, str]] = {}
                stream_usage: UsageStats | None = None

                for chunk in response_stream:
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

                function_calls: list[FunctionCall] = []
                tool_calls_raw: list[dict[str, Any]] = []
                for tc_dict in tool_calls_acc.values():
                    function_calls.append(
                        FunctionCall(
                            name=tc_dict["name"],
                            call_id=tc_dict["id"],
                            arguments=tc_dict["arguments"],
                        )
                    )
                    tool_calls_raw.append(
                        {
                            "id": tc_dict["id"],
                            "type": "function",
                            "function": {
                                "name": tc_dict["name"],
                                "arguments": tc_dict["arguments"],
                            },
                        }
                    )

                raw_output_item: dict[str, Any] = {"role": "assistant"}
                if accumulated_text:
                    raw_output_item["content"] = accumulated_text
                if tool_calls_raw:
                    raw_output_item["tool_calls"] = tool_calls_raw

                if not stream_usage or stream_usage.total_tokens == 0:
                    prompt_str = " ".join([str(m.get("content", "")) for m in messages])
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
                    raw_output=[raw_output_item],
                    usage=stream_usage,
                )

            # Non-streaming execution
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
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
            raise LLMError(f"API 速率限制 (Rate Limit): {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"API 调用失败: {exc}") from exc

        if not response.choices:
            return LLMResponse(text=None)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content
        function_calls = []
        tool_calls_raw = []

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                function_calls.append(
                    FunctionCall(
                        name=tc.function.name,
                        call_id=tc.id,
                        arguments=tc.function.arguments,
                    )
                )
                tool_calls_raw.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        raw_output_item = {"role": "assistant"}
        if text is not None:
            raw_output_item["content"] = text
        if tool_calls_raw:
            raw_output_item["tool_calls"] = tool_calls_raw

        resp_usage: UsageStats
        if getattr(response, "usage", None):
            u = response.usage
            resp_usage = UsageStats(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )
        else:
            prompt_str = " ".join([str(m.get("content", "")) for m in messages])
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
            raw_output=[raw_output_item],
            usage=resp_usage,
        )


# Backward compatibility alias
OpenAIResponsesClient = OpenAIChatCompletionsClient

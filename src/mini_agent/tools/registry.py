"""Tool registry: register, schema generation, and parse+validate+execute dispatch."""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from mini_agent.models import ToolResult
from mini_agent.tools.protocol import Tool, ToolContext


def _flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$ref so the root schema is a single JSON Schema object."""
    defs = dict(schema.get("$defs") or schema.get("definitions") or {})
    root = {k: v for k, v in schema.items() if k not in ("$defs", "definitions")}

    def resolve(node: Any, stack: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            name = ref.rsplit("/", 1)[-1]
            if name in stack:
                extras = {k: v for k, v in node.items() if k != "$ref"}
                return {k: resolve(v, stack) for k, v in extras.items()}
            target = defs.get(name)
            if target is None:
                return {k: resolve(v, stack) for k, v in node.items()}
            merged = copy.deepcopy(target)
            for key, value in node.items():
                if key != "$ref":
                    merged[key] = value
            return resolve(merged, stack | {name})
        return {k: resolve(v, stack) for k, v in node.items()}

    return resolve(root, frozenset())


def _parameters_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = _flatten_json_schema(copy.deepcopy(model.model_json_schema()))
    schema["additionalProperties"] = False
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    return schema


class ToolRegistry:
    """Named tool catalog. dispatch() does not check permissions."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        if not name:
            return None
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def json_schemas(self, *, send_strict: bool = False) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool in self.list():
            function: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": _parameters_schema(tool.input_model),
            }
            if send_strict:
                function["strict"] = True
            schemas.append({"type": "function", "function": function})
        return schemas

    def dispatch(self, name: str, raw_arguments: str, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, content="", error=f"未知的工具名称: '{name}'")

        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"工具参数不是合法的 JSON 字符串: {exc}",
            )

        if not isinstance(args, dict):
            return ToolResult(
                ok=False,
                content="",
                error=f"工具参数必须为 JSON 对象 (dict)，收到: {type(args).__name__}",
            )

        try:
            inp = tool.input_model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"{name} 参数校验失败: {exc}",
            )

        return tool.execute(inp, ctx)

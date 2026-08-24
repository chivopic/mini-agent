"""Legacy dict-history compaction. Token-aware windowing lives in compaction.py."""

import json
from typing import Any


def compact_history(
    history: list[dict[str, Any]],
    max_recent_messages: int = 8,
    max_tool_output_chars: int = 250,
) -> list[dict[str, Any]]:
    """Compact older tool outputs in history to save context window and tokens.

    Preserves:
    - System message (always)
    - The most recent `max_recent_messages` items (unmodified)
    Compacts:
    - Historical tool outputs before the recent window if they exceed `max_tool_output_chars`.
    """
    if len(history) <= max_recent_messages + 1:
        return list(history)

    compacted: list[dict[str, Any]] = []
    cutoff_index = len(history) - max_recent_messages

    for i, item in enumerate(history):
        # Always preserve system prompt and recent window intact
        if i == 0 or i >= cutoff_index:
            compacted.append(item)
            continue

        item_type = item.get("type")
        role = item.get("role")

        # Compact historical function_call_output
        if item_type == "function_call_output":
            raw_output = item.get("output", "")
            if isinstance(raw_output, str) and len(raw_output) > max_tool_output_chars:
                try:
                    data = json.loads(raw_output)
                    if isinstance(data, dict) and "content" in data:
                        orig_len = len(str(data["content"]))
                        data["content"] = f"[前序轮次执行成功，输出已折叠，原长 {orig_len} 字符]"
                        compacted_output = json.dumps(data, ensure_ascii=False)
                    else:
                        compacted_output = f"[历史工具结果已折叠，原长 {len(raw_output)} 字符]"
                except Exception:
                    compacted_output = f"[历史工具输出已折叠，原长 {len(raw_output)} 字符]"

                compacted.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("call_id", ""),
                        "output": compacted_output,
                    }
                )
            else:
                compacted.append(item)
            continue

        # Compact historical tool message
        if role == "tool":
            content = str(item.get("content", ""))
            if len(content) > max_tool_output_chars:
                compacted.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("tool_call_id", ""),
                        "content": f"[历史工具输出已折叠，原长 {len(content)} 字符]",
                    }
                )
            else:
                compacted.append(item)
            continue

        compacted.append(item)

    return compacted

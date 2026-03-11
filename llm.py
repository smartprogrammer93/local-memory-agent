"""
LLMAgent — Agentic tool-calling loop over OpenAI-compatible API.

Wraps AsyncOpenAI with automatic tool schema generation from typed
Python functions, and a multi-round tool-calling loop.
"""

import inspect
import json
import typing
from collections.abc import Callable
from typing import Any, Optional, get_args, get_origin

from openai import AsyncOpenAI

# ─── Type mapping ────────────────────────────────────────────

_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

MAX_TOOL_ROUNDS = 10


def _json_type(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema snippet."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    # Handle Optional[X] → nullable X
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Union and type(None) in args:
        # Optional[X] — pick the non-None type
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"type": "string"}

    if origin in (list, typing.List):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _json_type(args[0])
        return schema

    if origin in (dict, typing.Dict):
        return {"type": "object"}

    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}

    return {"type": "string"}


def _parse_arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Parse 'Args:' section from a Google-style docstring."""
    if not docstring:
        return {}

    lines = docstring.split("\n")
    descriptions: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            # End of Args section on next top-level section or blank after content
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                break
            if stripped.lower().startswith(("returns:", "raises:", "yields:", "note:", "example")):
                break
            if ":" in stripped:
                param_part, _, desc = stripped.partition(":")
                param_name = param_part.strip().split("(")[0].strip()
                if param_name:
                    descriptions[param_name] = desc.strip()
    return descriptions


def _fn_to_schema(fn: Callable) -> dict:
    """Auto-generate an OpenAI tool schema from a Python function.

    Uses the function's name, docstring (parses Args: section),
    and type annotations to build the schema.
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].strip() if doc else fn.__name__
    arg_descs = _parse_arg_descriptions(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        prop = _json_type(param.annotation)
        if name in arg_descs:
            prop["description"] = arg_descs[name]
        properties[name] = prop

        # Required if no default value and not Optional
        origin = get_origin(param.annotation)
        args = get_args(param.annotation)
        is_optional = origin is typing.Union and type(None) in args
        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ─── LLMAgent ───────────────────────────────────────────────


class LLMAgent:
    """Agentic loop that calls tools via OpenAI-compatible chat completions."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Callable],
        client: AsyncOpenAI,
        model: str,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.model = model

        # Map function name → callable for dispatch
        self._tool_fns: dict[str, Callable] = {fn.__name__: fn for fn in tools}
        self._tool_schemas: list[dict] = [_fn_to_schema(fn) for fn in tools]

    async def run(self, message: str | list) -> str:
        """Run the agent with a user message (string or pre-built message list)."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        else:
            messages.extend(message)

        return await self._loop(messages)

    async def _loop(self, messages: list[dict]) -> str:
        """Core tool-calling loop. Calls the model, executes tools, repeats."""
        for _ in range(MAX_TOOL_ROUNDS):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if self._tool_schemas:
                kwargs["tools"] = self._tool_schemas
                kwargs["tool_choice"] = "auto"

            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }

            response = await self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # Append assistant message with tool calls
            messages.append(msg.model_dump())

            # Execute each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn = self._tool_fns.get(fn_name)
                if fn is None:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    try:
                        raw_args = tc.function.arguments
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        if inspect.iscoroutinefunction(fn):
                            result_obj = await fn(**args)
                        else:
                            result_obj = fn(**args)
                        result = json.dumps(result_obj, default=str)
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Max tool rounds reached."

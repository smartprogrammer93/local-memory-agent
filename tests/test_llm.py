"""Tests for the LLM module — schema generation, type mapping, and tool loop."""
import inspect
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm import (
    MAX_TOOL_ROUNDS,
    LLMAgent,
    _fn_to_schema,
    _json_type,
    _parse_arg_descriptions,
)

# ─── _json_type tests ─────────────────────────────────────────


def test_json_type_basic():
    assert _json_type(str) == {"type": "string"}
    assert _json_type(int) == {"type": "integer"}
    assert _json_type(float) == {"type": "number"}
    assert _json_type(bool) == {"type": "boolean"}
    assert _json_type(list) == {"type": "array"}
    assert _json_type(dict) == {"type": "object"}


def test_json_type_list_with_item():
    assert _json_type(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert _json_type(list[int]) == {"type": "array", "items": {"type": "integer"}}


def test_json_type_optional():
    assert _json_type(Optional[str]) == {"type": "string"}
    assert _json_type(Optional[int]) == {"type": "integer"}
    assert _json_type(Optional[bool]) == {"type": "boolean"}


def test_json_type_empty():
    assert _json_type(inspect.Parameter.empty) == {"type": "string"}


# ─── _parse_arg_descriptions tests ────────────────────────────


def test_parse_arg_descriptions():
    doc = """Do something.

    Args:
        name: The person's name.
        age: Their age in years.

    Returns:
        A greeting string.
    """
    descs = _parse_arg_descriptions(doc)
    assert descs["name"] == "The person's name."
    assert descs["age"] == "Their age in years."


def test_parse_arg_descriptions_empty():
    assert _parse_arg_descriptions(None) == {}
    assert _parse_arg_descriptions("") == {}
    assert _parse_arg_descriptions("No args section here.") == {}


# ─── _fn_to_schema tests ─────────────────────────────────────


def test_fn_to_schema_basic():
    def greet(name: str, loud: bool = False) -> str:
        """Say hello to someone.

        Args:
            name: Who to greet.
            loud: Whether to shout.
        """
        return f"Hello {name}"

    schema = _fn_to_schema(greet)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "greet"
    assert fn["description"] == "Say hello to someone."
    props = fn["parameters"]["properties"]
    assert props["name"] == {"type": "string", "description": "Who to greet."}
    assert props["loud"] == {"type": "boolean", "description": "Whether to shout."}
    assert fn["parameters"]["required"] == ["name"]


def test_fn_to_schema_int_and_optional():
    def calculate(x: int, y: int, label: Optional[str] = None) -> int:
        """Add two numbers.

        Args:
            x: First number.
            y: Second number.
            label: Optional label for the result.
        """
        return x + y

    schema = _fn_to_schema(calculate)
    props = schema["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "integer"
    assert props["label"]["type"] == "string"
    required = schema["function"]["parameters"]["required"]
    assert "x" in required
    assert "y" in required
    assert "label" not in required


def test_fn_to_schema_list_param():
    def process(items: list[str], count: int) -> dict:
        """Process items.

        Args:
            items: List of strings to process.
            count: How many to take.
        """
        return {}

    schema = _fn_to_schema(process)
    props = schema["function"]["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert props["items"]["items"] == {"type": "string"}
    assert props["count"]["type"] == "integer"


def test_fn_to_schema_no_args():
    def get_time() -> str:
        """Return current time."""
        return "12:00"

    schema = _fn_to_schema(get_time)
    fn = schema["function"]
    assert fn["name"] == "get_time"
    assert fn["description"] == "Return current time."
    assert fn["parameters"]["properties"] == {}
    assert fn["parameters"]["required"] == []


# ─── Helpers for mocking the OpenAI client ────────────────────


def _make_tool_call(call_id: str, fn_name: str, arguments: dict):
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = fn_name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_response(content: str | None = None, tool_calls: list | None = None):
    """Build a mock chat completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    # model_dump for appending assistant message
    dump = {"role": "assistant", "content": content}
    if tool_calls:
        dump["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    msg.model_dump.return_value = dump

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_client(responses: list):
    """Build a mock AsyncOpenAI client that returns responses in order."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


# ─── LLMAgent._loop tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_loop_text_only_response():
    """Model returns text with no tool calls — should return immediately."""
    client = _make_client([_make_response(content="Hello there!")])

    agent = LLMAgent(
        name="test",
        system_prompt="You are helpful.",
        tools=[],
        client=client,
        model="test-model",
    )

    result = await agent.run("Hi")
    assert result == "Hello there!"
    assert client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_loop_tool_call_then_text():
    """Model calls a tool, gets result, then returns text."""
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    tc = _make_tool_call("call_1", "add", {"a": 3, "b": 4})
    responses = [
        _make_response(tool_calls=[tc]),
        _make_response(content="The sum is 7."),
    ]
    client = _make_client(responses)

    agent = LLMAgent(
        name="test",
        system_prompt="You are a calculator.",
        tools=[add],
        client=client,
        model="test-model",
    )

    result = await agent.run("What is 3 + 4?")
    assert result == "The sum is 7."
    assert client.chat.completions.create.call_count == 2

    # Verify tool result was sent back in messages
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert tool_msg[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_msg[0]["content"]) == 7


@pytest.mark.asyncio
async def test_loop_async_tool():
    """Model calls an async tool function."""
    async def fetch_data(url: str) -> str:
        """Fetch data from a URL.

        Args:
            url: The URL to fetch.
        """
        return f"data from {url}"

    tc = _make_tool_call("call_2", "fetch_data", {"url": "http://example.com"})
    responses = [
        _make_response(tool_calls=[tc]),
        _make_response(content="Got the data."),
    ]
    client = _make_client(responses)

    agent = LLMAgent(
        name="test",
        system_prompt="You fetch data.",
        tools=[fetch_data],
        client=client,
        model="test-model",
    )

    result = await agent.run("Fetch example.com")
    assert result == "Got the data."

    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
    assert json.loads(tool_msg[0]["content"]) == "data from http://example.com"


@pytest.mark.asyncio
async def test_loop_max_rounds():
    """Model always returns tool calls — should hit max rounds limit."""
    def noop() -> str:
        """Do nothing."""
        return "ok"

    tc = _make_tool_call("call_loop", "noop", {})
    # Return a tool call every round
    responses = [_make_response(tool_calls=[tc]) for _ in range(MAX_TOOL_ROUNDS)]
    client = _make_client(responses)

    agent = LLMAgent(
        name="test",
        system_prompt="Loop forever.",
        tools=[noop],
        client=client,
        model="test-model",
    )

    result = await agent.run("Go")
    assert result == "Max tool rounds reached."
    assert client.chat.completions.create.call_count == MAX_TOOL_ROUNDS


@pytest.mark.asyncio
async def test_loop_unknown_tool():
    """Model calls a tool that doesn't exist — error is sent back, loop continues."""
    tc_unknown = _make_tool_call("call_bad", "nonexistent_tool", {"x": 1})
    responses = [
        _make_response(tool_calls=[tc_unknown]),
        _make_response(content="I see the error, let me try differently."),
    ]
    client = _make_client(responses)

    agent = LLMAgent(
        name="test",
        system_prompt="You are helpful.",
        tools=[],  # no tools registered
        client=client,
        model="test-model",
    )

    result = await agent.run("Call something")
    assert result == "I see the error, let me try differently."

    # Verify error was sent back as tool result
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    error_data = json.loads(tool_msg[0]["content"])
    assert "error" in error_data
    assert "nonexistent_tool" in error_data["error"]


@pytest.mark.asyncio
async def test_loop_tool_exception():
    """Tool function raises an exception — error is captured and sent back."""
    def bad_tool(x: int) -> str:
        """A tool that fails.

        Args:
            x: Some input.
        """
        raise ValueError("something broke")

    tc = _make_tool_call("call_err", "bad_tool", {"x": 42})
    responses = [
        _make_response(tool_calls=[tc]),
        _make_response(content="Tool failed, sorry."),
    ]
    client = _make_client(responses)

    agent = LLMAgent(
        name="test",
        system_prompt="Test.",
        tools=[bad_tool],
        client=client,
        model="test-model",
    )

    result = await agent.run("Do it")
    assert result == "Tool failed, sorry."

    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
    error_data = json.loads(tool_msg[0]["content"])
    assert "error" in error_data
    assert "something broke" in error_data["error"]

"""Tests for agents.py — build_agents and MemoryOrchestrator routing."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

# Redirect DB to a temp file before any tools import touches the real DB
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ.setdefault("MEMORY_DB", _tmp_db.name)

from agents import (
    CONSOLIDATE_PROMPT,
    INGEST_PROMPT,
    QUERY_PROMPT,
    MemoryOrchestrator,
    build_agents,
)
from llm import LLMAgent
from tools import (
    read_all_memories,
    read_consolidation_history,
    read_unconsolidated_memories,
    store_consolidation,
    store_memory,
)

# ─── build_agents tests ──────────────────────────────────────


def test_build_agents_returns_expected_keys():
    """build_agents returns dict with exactly 'ingest', 'consolidate', 'query'."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    assert set(agents.keys()) == {"ingest", "consolidate", "query"}


def test_build_agents_returns_llm_agent_instances():
    """Each value is a LLMAgent."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    for name, agent in agents.items():
        assert isinstance(agent, LLMAgent), f"agents['{name}'] is not a LLMAgent"


def test_build_agents_ingest_has_correct_tools():
    """Ingest agent has store_memory tool."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    ingest = agents["ingest"]
    assert "store_memory" in ingest._tool_fns
    assert len(ingest._tool_fns) == 1


def test_build_agents_consolidate_has_correct_tools():
    """Consolidate agent has read_unconsolidated_memories and store_consolidation."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    consolidate = agents["consolidate"]
    assert "read_unconsolidated_memories" in consolidate._tool_fns
    assert "store_consolidation" in consolidate._tool_fns
    assert len(consolidate._tool_fns) == 2


def test_build_agents_query_has_correct_tools():
    """Query agent has read_all_memories and read_consolidation_history."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    query = agents["query"]
    assert "read_all_memories" in query._tool_fns
    assert "read_consolidation_history" in query._tool_fns
    assert len(query._tool_fns) == 2


def test_build_agents_correct_prompts():
    """Each agent has the correct system prompt."""
    client = AsyncMock()
    agents = build_agents(client, "test-model")

    assert agents["ingest"].system_prompt == INGEST_PROMPT
    assert agents["consolidate"].system_prompt == CONSOLIDATE_PROMPT
    assert agents["query"].system_prompt == QUERY_PROMPT


# ─── Helpers ──────────────────────────────────────────────────


def _make_orchestrator() -> tuple[MemoryOrchestrator, dict[str, AsyncMock]]:
    """Create an orchestrator with mocked agent .run() methods."""
    mocks: dict[str, AsyncMock] = {}
    agents: dict[str, MagicMock] = {}

    for name in ("ingest", "consolidate", "query"):
        agent = MagicMock(spec=LLMAgent)
        agent.run = AsyncMock(return_value=f"{name} response")
        agents[name] = agent
        mocks[name] = agent.run

    client = AsyncMock()
    orch = MemoryOrchestrator(agents=agents, client=client, model="test-model")
    return orch, mocks


# ─── MemoryOrchestrator.route tests ──────────────────────────


@pytest.mark.asyncio
async def test_route_consolidate_keyword():
    """'consolidate' in message routes to consolidate agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("consolidate my memories")
    assert result == "consolidate response"
    mocks["consolidate"].assert_awaited_once()
    mocks["ingest"].assert_not_awaited()
    mocks["query"].assert_not_awaited()


@pytest.mark.asyncio
async def test_route_consolidate_merge_keyword():
    """'merge' keyword routes to consolidate agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("merge all my notes together")
    assert result == "consolidate response"
    mocks["consolidate"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_consolidate_patterns_keyword():
    """'patterns' keyword routes to consolidate agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("find patterns in my data")
    assert result == "consolidate response"
    mocks["consolidate"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_ingest_remember_keyword():
    """'remember' keyword routes to ingest agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("remember this fact")
    assert result == "ingest response"
    mocks["ingest"].assert_awaited_once()
    mocks["consolidate"].assert_not_awaited()
    mocks["query"].assert_not_awaited()


@pytest.mark.asyncio
async def test_route_ingest_store_keyword():
    """'store' keyword routes to ingest agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("store this information")
    assert result == "ingest response"
    mocks["ingest"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_ingest_analyze_keyword():
    """'analyze' keyword routes to ingest agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("analyze this document")
    assert result == "ingest response"
    mocks["ingest"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_query_what_keyword():
    """'what' keyword routes to query agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("what do I know about X")
    assert result == "query response"
    mocks["query"].assert_awaited_once()
    mocks["ingest"].assert_not_awaited()
    mocks["consolidate"].assert_not_awaited()


@pytest.mark.asyncio
async def test_route_query_tell_me_keyword():
    """'tell me' keyword routes to query agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("tell me about Python")
    assert result == "query response"
    mocks["query"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_query_based_on_keyword():
    """'based on' keyword routes to query agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("based on my notes, summarize")
    assert result == "query response"
    mocks["query"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_fallback_to_query():
    """Unknown message with no keywords falls back to query agent."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("hello there, how are you?")
    assert result == "query response"
    mocks["query"].assert_awaited_once()
    mocks["ingest"].assert_not_awaited()
    mocks["consolidate"].assert_not_awaited()


@pytest.mark.asyncio
async def test_route_ingest_with_content():
    """When content is provided, ingest agent receives the content, not the message."""
    orch, mocks = _make_orchestrator()

    await orch.route("remember this fact", content="The sky is blue.")
    mocks["ingest"].assert_awaited_once()
    call_arg = mocks["ingest"].call_args[0][0]
    assert "The sky is blue." in call_arg


@pytest.mark.asyncio
async def test_route_ingest_without_content_uses_message():
    """When no content is provided, ingest agent receives the message itself."""
    orch, mocks = _make_orchestrator()

    await orch.route("remember that cats are great")
    call_arg = mocks["ingest"].call_args[0][0]
    assert "remember that cats are great" in call_arg


@pytest.mark.asyncio
async def test_route_case_insensitive():
    """Routing is case-insensitive."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("CONSOLIDATE everything")
    assert result == "consolidate response"
    mocks["consolidate"].assert_awaited_once()


@pytest.mark.asyncio
async def test_route_consolidate_takes_priority_over_ingest():
    """When both consolidate and ingest keywords present, consolidate wins."""
    orch, mocks = _make_orchestrator()

    result = await orch.route("consolidate and store my memories")
    assert result == "consolidate response"
    mocks["consolidate"].assert_awaited_once()
    mocks["ingest"].assert_not_awaited()

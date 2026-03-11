"""Tests for HTTP API endpoints in agent.py — build_http."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from agent import build_http, MemoryAgent


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_agent():
    """Create a MemoryAgent with all async methods mocked."""
    agent = MagicMock(spec=MemoryAgent)
    agent.query = AsyncMock(return_value="mocked query answer")
    agent.ingest = AsyncMock(return_value="mocked ingest result")
    agent.ingest_file = AsyncMock(return_value="mocked file ingest result")
    agent.consolidate = AsyncMock(return_value="mocked consolidation result")
    agent.run = AsyncMock(return_value="mocked run result")
    return agent


@pytest.fixture
def watch_path(tmp_path):
    """Stable tmp_path shared between client fixture and tests."""
    return tmp_path / "inbox"


@pytest_asyncio.fixture
async def client(mock_agent, watch_path):
    """Create a test client using aiohttp.test_utils directly."""
    app = build_http(mock_agent, watch_path=str(watch_path))
    async with TestClient(TestServer(app)) as c:
        yield c


# ─── GET /health ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"status": "ok"}


# ─── POST /ingest ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_success(client, mock_agent):
    resp = await client.post("/ingest", json={"text": "hello world", "source": "test"})
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ingested"
    assert data["response"] == "mocked ingest result"
    mock_agent.ingest.assert_awaited_once_with("hello world", source="test")


@pytest.mark.asyncio
async def test_ingest_default_source(client, mock_agent):
    resp = await client.post("/ingest", json={"text": "some data"})
    assert resp.status == 200
    mock_agent.ingest.assert_awaited_once_with("some data", source="api")


@pytest.mark.asyncio
async def test_ingest_missing_text(client):
    resp = await client.post("/ingest", json={"source": "test"})
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "text" in data["error"].lower()


@pytest.mark.asyncio
async def test_ingest_empty_text(client):
    resp = await client.post("/ingest", json={"text": "   ", "source": "test"})
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_ingest_invalid_json(client):
    resp = await client.post(
        "/ingest", data=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


# ─── GET /query ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_success(client, mock_agent):
    resp = await client.get("/query", params={"q": "test question"})
    assert resp.status == 200
    data = await resp.json()
    assert data["question"] == "test question"
    assert data["answer"] == "mocked query answer"
    mock_agent.query.assert_awaited_once_with("test question")


@pytest.mark.asyncio
async def test_query_missing_param(client):
    resp = await client.get("/query")
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "q" in data["error"].lower()


@pytest.mark.asyncio
async def test_query_empty_param(client):
    resp = await client.get("/query", params={"q": "   "})
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


# ─── POST /consolidate ────────────────────────────────────────


@pytest.mark.asyncio
async def test_consolidate_success(client, mock_agent):
    resp = await client.post("/consolidate")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "done"
    assert data["response"] == "mocked consolidation result"
    mock_agent.consolidate.assert_awaited_once()


# ─── POST /ingest-file ────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_file_success(client, mock_agent, watch_path):
    form = aiohttp.FormData()
    form.add_field(
        "file", io.BytesIO(b"file content here"),
        filename="test.txt", content_type="text/plain",
    )

    resp = await client.post("/ingest-file", data=form)
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ingested"
    assert body["filename"] == "test.txt"
    assert body["response"] == "mocked file ingest result"
    mock_agent.ingest_file.assert_awaited_once()

    # Verify file was written to the watch path
    written = watch_path / "test.txt"
    assert written.exists()
    assert written.read_bytes() == b"file content here"


@pytest.mark.asyncio
async def test_ingest_file_missing_field(client):
    form = aiohttp.FormData()
    form.add_field("wrong_name", io.BytesIO(b"data"), filename="test.txt")

    resp = await client.post("/ingest-file", data=form)
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_ingest_file_agent_error(client, mock_agent):
    """When agent.ingest_file raises, endpoint returns 500."""
    mock_agent.ingest_file.side_effect = RuntimeError("processing failed")

    form = aiohttp.FormData()
    form.add_field(
        "file", io.BytesIO(b"data"), filename="bad.txt", content_type="text/plain",
    )

    resp = await client.post("/ingest-file", data=form)
    assert resp.status == 500
    data = await resp.json()
    assert "error" in data
    assert "processing failed" in data["error"]

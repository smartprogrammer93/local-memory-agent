"""Tests for database operations and tool functions."""
import json
import os
import tempfile

import pytest

# Point DB to a temp file before importing agent
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["MEMORY_DB"] = _tmp.name

from tools import (
    _get_db as get_db,
    store_memory,
    read_all_memories,
    read_unconsolidated_memories,
    store_consolidation,
    read_consolidation_history,
    get_memory_stats,
    delete_memory,
    clear_all_memories,
)


@pytest.fixture(autouse=True)
def clean_db():
    """Reset DB before each test."""
    db = get_db()
    db.execute("DELETE FROM memories")
    db.execute("DELETE FROM consolidations")
    db.execute("DELETE FROM processed_files")
    db.commit()
    db.close()
    yield


def test_store_and_read_memory():
    result = store_memory(
        raw_text="Alice works at Acme Corp",
        summary="Alice is an Acme employee",
        entities=["Alice", "Acme Corp"],
        topics=["work", "people"],
        importance=0.7,
        source="test",
    )
    assert result["status"] == "stored"
    assert result["memory_id"] >= 1

    data = read_all_memories()
    assert data["count"] == 1
    assert data["memories"][0]["summary"] == "Alice is an Acme employee"
    assert data["memories"][0]["source"] == "test"
    assert data["memories"][0]["entities"] == ["Alice", "Acme Corp"]


def test_read_unconsolidated():
    store_memory("a", "s1", ["e"], ["t"], 0.5)
    store_memory("b", "s2", ["e"], ["t"], 0.5)

    data = read_unconsolidated_memories()
    assert data["count"] == 2


def test_store_consolidation():
    r1 = store_memory("a", "s1", [], [], 0.5)
    r2 = store_memory("b", "s2", [], [], 0.5)

    result = store_consolidation(
        source_ids=[r1["memory_id"], r2["memory_id"]],
        summary="Combined summary",
        insight="Key insight",
        connections=[{
            "from_id": r1["memory_id"],
            "to_id": r2["memory_id"],
            "relationship": "related",
        }],
    )
    assert result["status"] == "consolidated"
    assert result["memories_processed"] == 2

    # Memories should be marked consolidated
    data = read_unconsolidated_memories()
    assert data["count"] == 0

    history = read_consolidation_history()
    assert history["count"] == 1
    assert history["consolidations"][0]["insight"] == "Key insight"


def test_get_memory_stats():
    store_memory("a", "s", [], [], 0.5)
    store_memory("b", "s", [], [], 0.5)

    stats = get_memory_stats()
    assert stats["total_memories"] == 2
    assert stats["unconsolidated"] == 2
    assert stats["consolidations"] == 0


def test_delete_memory():
    r = store_memory("a", "s", [], [], 0.5)
    mid = r["memory_id"]

    result = delete_memory(mid)
    assert result["status"] == "deleted"

    data = read_all_memories()
    assert data["count"] == 0


def test_delete_memory_not_found():
    result = delete_memory(9999)
    assert result["status"] == "not_found"


def test_clear_all_memories():
    store_memory("a", "s", [], [], 0.5)
    store_memory("b", "s", [], [], 0.5)

    result = clear_all_memories()
    assert result["status"] == "cleared"
    assert result["memories_deleted"] == 2

    data = read_all_memories()
    assert data["count"] == 0

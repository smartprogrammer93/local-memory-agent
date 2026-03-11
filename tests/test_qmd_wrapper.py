"""Tests for qmd_wrapper.py — CLI interface for Local Memory Agent."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qmd_wrapper


# ─── Helpers ───────────────────────────────────────────────────


def _make_results(n, base_score=0.95):
    """Generate n fake search results with descending scores."""
    results = []
    for i in range(n):
        results.append(
            {
                "docid": f"doc-{i}",
                "filepath": f"/data/file{i}.txt",
                "title": f"Result {i}",
                "score": round(base_score - i * 0.1, 2),
                "snippet": f"Content snippet {i}",
            }
        )
    return results


def _run_cli(args):
    """Parse args and invoke the subcommand handler; returns nothing."""
    parser = qmd_wrapper.build_parser()
    parsed = parser.parse_args(args)
    parsed.func(parsed)


# ─── Tests ─────────────────────────────────────────────────────


class TestSearchFormatsOutputCorrectly:
    """test_search_formats_output_correctly"""

    @patch("qmd_wrapper.requests.get")
    def test_search_formats_output_correctly(self, mock_get, capsys):
        results = _make_results(2)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "test", "query"])

        out = capsys.readouterr().out
        # Verify qmd:// URIs
        assert "qmd:///data/file0.txt" in out
        assert "qmd:///data/file1.txt" in out
        # Verify titles
        assert "Title: Result 0" in out
        assert "Title: Result 1" in out
        # Verify scores as percentages
        assert "Score: 95%" in out
        assert "Score: 85%" in out
        # Verify content snippets
        assert "Content snippet 0" in out
        assert "Content snippet 1" in out
        # Verify separator between results
        assert "---" in out


class TestSearchNFlag:
    """test_search_n_flag"""

    @patch("qmd_wrapper.requests.get")
    def test_search_n_flag(self, mock_get, capsys):
        results = _make_results(5)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "-n", "2", "test", "query"])

        out = capsys.readouterr().out
        # Should only have 2 results, so only 1 separator
        assert out.count("Title:") == 2
        assert "Result 0" in out
        assert "Result 1" in out
        assert "Result 2" not in out


class TestSearchFilesFlag:
    """test_search_files_flag"""

    @patch("qmd_wrapper.requests.get")
    def test_search_files_flag(self, mock_get, capsys):
        results = _make_results(2)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--files", "test", "query"])

        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l]
        assert len(lines) == 2
        # CSV format: docid,score,filepath
        assert lines[0] == "doc-0,95,/data/file0.txt"
        assert lines[1] == "doc-1,85,/data/file1.txt"


class TestGetReadsRealFile:
    """test_get_reads_real_file"""

    def test_get_reads_real_file(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nsecond line\n")
            path = f.name
        try:
            _run_cli(["get", path])
            out = capsys.readouterr().out
            assert out == "hello world\nsecond line\n"
        finally:
            os.unlink(path)


class TestGetWithLineRange:
    """test_get_with_line_range"""

    def test_get_with_line_range(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(1, 21):
                f.write(f"line {i}\n")
            path = f.name
        try:
            _run_cli(["get", "-l", "5", "--from", "10", path])
            out = capsys.readouterr().out
            lines = out.splitlines()
            assert len(lines) == 5
            assert lines[0] == "line 10"
            assert lines[4] == "line 14"
        finally:
            os.unlink(path)


class TestStatusFormatsCorrectly:
    """test_status_formats_correctly"""

    @patch("qmd_wrapper.requests.get")
    def test_status_formats_correctly(self, mock_get, capsys):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"total_memories": 42, "last_updated": "2026-03-11T10:00:00Z"},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["status"])

        out = capsys.readouterr().out
        assert "QMD Status" in out
        assert "Documents: 42" in out
        assert "Last updated: 2026-03-11T10:00:00Z" in out


class TestAgentUnreachableReturnsEmpty:
    """test_agent_unreachable_returns_empty"""

    @patch("qmd_wrapper.requests.get")
    def test_agent_unreachable_returns_empty(self, mock_get, capsys):
        mock_get.side_effect = requests.ConnectionError("refused")

        # Should not raise — exit code 0
        _run_cli(["search", "test"])

        out = capsys.readouterr().out
        # stdout should be empty (error goes to stderr)
        assert out.strip() == ""


class TestUpdateNoop:
    """test_update_noop"""

    def test_update_noop(self, capsys):
        _run_cli(["update"])

        out = capsys.readouterr().out
        assert "no-op" in out.lower() or "update" in out.lower()


class TestVsearchSameAsSearch:
    """test_vsearch_same_as_search"""

    @patch("qmd_wrapper.requests.get")
    def test_vsearch_same_as_search(self, mock_get, capsys):
        results = _make_results(1)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["vsearch", "test"])

        # vsearch uses the semantic /query endpoint
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "/query" in call_url

        out = capsys.readouterr().out
        assert "Title: Result 0" in out


class TestMinScoreFilter:
    """test_min_score_filter"""

    @patch("qmd_wrapper.requests.get")
    def test_min_score_filter(self, mock_get, capsys):
        results = [
            {"docid": "a", "filepath": "/a.txt", "title": "High", "score": 0.95, "snippet": "high"},
            {"docid": "b", "filepath": "/b.txt", "title": "Medium", "score": 0.60, "snippet": "medium"},
            {"docid": "c", "filepath": "/c.txt", "title": "Low", "score": 0.30, "snippet": "low"},
        ]
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--min-score", "50", "test"])

        out = capsys.readouterr().out
        assert out.count("Title:") == 2
        assert "High" in out
        assert "Medium" in out
        assert "Low" not in out


# ─── New tests for --json / -c / _get_qmd_anchor_docid ────────


class TestSearchUsesKeywordEndpoint:
    """search and query subcommands use fast /search; vsearch uses LLM /query."""

    @patch("qmd_wrapper.requests.get")
    def test_search_uses_search_endpoint(self, mock_get, capsys):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": []})
        mock_get.return_value.raise_for_status = MagicMock()
        _run_cli(["search", "test"])
        assert "/search" in mock_get.call_args[0][0]

    @patch("qmd_wrapper.requests.get")
    def test_query_uses_search_endpoint(self, mock_get, capsys):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": []})
        mock_get.return_value.raise_for_status = MagicMock()
        _run_cli(["query", "test"])
        assert "/search" in mock_get.call_args[0][0]

    @patch("qmd_wrapper.requests.get")
    def test_vsearch_uses_query_endpoint(self, mock_get, capsys):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"answer": "result"})
        mock_get.return_value.raise_for_status = MagicMock()
        _run_cli(["vsearch", "test"])
        assert "/query" in mock_get.call_args[0][0]


class TestSearchJsonFlag:
    """--json flag returns valid JSON array with docid/score/snippet fields."""

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="deadbeef123")
    @patch("qmd_wrapper.requests.get")
    def test_json_output_is_valid_array(self, mock_get, mock_docid, capsys):
        import json
        results = _make_results(2)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--json", "test", "query"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="deadbeef123")
    @patch("qmd_wrapper.requests.get")
    def test_json_output_has_required_fields(self, mock_get, mock_docid, capsys):
        import json
        results = _make_results(1)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--json", "test"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        item = parsed[0]
        assert "docid" in item
        assert "score" in item
        assert "snippet" in item

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="deadbeef123")
    @patch("qmd_wrapper.requests.get")
    def test_json_output_uses_anchor_docid(self, mock_get, mock_docid, capsys):
        import json
        results = _make_results(1)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--json", "test"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed[0]["docid"] == "deadbeef123"

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="deadbeef123")
    @patch("qmd_wrapper.requests.get")
    def test_json_output_wraps_answer_field(self, mock_get, mock_docid, capsys):
        """When agent returns {answer: ...} with no results, wraps it as one item."""
        import json
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"answer": "Ahmad is Master in Kuwait"}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--json", "who is Ahmad"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert len(parsed) == 1
        assert "Ahmad is Master in Kuwait" in parsed[0]["snippet"]

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="deadbeef123")
    @patch("qmd_wrapper.requests.get")
    def test_json_output_unreachable_returns_empty_array(self, mock_get, mock_docid, capsys):
        import json
        mock_get.side_effect = requests.ConnectionError("refused")

        _run_cli(["search", "--json", "test"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == []


class TestCollectionFlag:
    """-c / --collection flag is accepted and ignored (searches all memories)."""

    @patch("qmd_wrapper.requests.get")
    def test_collection_flag_accepted(self, mock_get, capsys):
        results = _make_results(1)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        # Should not raise an error
        _run_cli(["search", "-c", "memory-root-main", "test"])

        out = capsys.readouterr().out
        assert "Result 0" in out

    @patch("qmd_wrapper.requests.get")
    def test_collection_long_flag_accepted(self, mock_get, capsys):
        results = _make_results(1)
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": results}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "--collection", "my-collection", "test"])

        out = capsys.readouterr().out
        assert "Result 0" in out

    @patch("qmd_wrapper._get_qmd_anchor_docid", return_value="abc")
    @patch("qmd_wrapper.requests.get")
    def test_json_and_collection_flags_together(self, mock_get, mock_docid, capsys):
        """Exact args OpenClaw passes: search <q> --json -n 6 -c memory-root-main"""
        import json
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"answer": "synthesized"}
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _run_cli(["search", "who is Ahmad", "--json", "-n", "6", "-c", "memory-root-main"])

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["snippet"] == "synthesized"


class TestGetQmdAnchorDocid:
    """_get_qmd_anchor_docid() returns a string in all cases."""

    def test_returns_string_when_db_missing(self):
        with patch("qmd_wrapper.QMD_INDEX", "/nonexistent/path/index.sqlite"):
            result = qmd_wrapper._get_qmd_anchor_docid()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_hash_from_real_db(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "index.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (hash TEXT, path TEXT, active INTEGER)")
        conn.execute("INSERT INTO documents VALUES ('abc123hash', 'memory.md', 1)")
        conn.commit()
        conn.close()

        with patch("qmd_wrapper.QMD_INDEX", db_path):
            result = qmd_wrapper._get_qmd_anchor_docid()
        assert result == "abc123hash"

    def test_returns_fallback_when_db_empty(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "index.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (hash TEXT, path TEXT, active INTEGER)")
        conn.commit()
        conn.close()

        with patch("qmd_wrapper.QMD_INDEX", db_path):
            result = qmd_wrapper._get_qmd_anchor_docid()
        assert isinstance(result, str)
        assert len(result) > 0


# ─── Tests for query mode (LLM expansion + SQLite search) ────


class TestQueryModeExpansion:
    """query mode calls /expand then /search; search/vsearch do not call /expand."""

    @patch("qmd_wrapper.requests.get")
    def test_query_mode_hits_expand_then_search(self, mock_get, capsys):
        """query subcommand calls GET /expand then GET /search (two calls)."""
        expand_resp = MagicMock(status_code=200, json=lambda: {"expanded": "cat feline kitten"})
        expand_resp.raise_for_status = MagicMock()
        search_resp = MagicMock(status_code=200, json=lambda: {"results": _make_results(1)})
        search_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [expand_resp, search_resp]

        _run_cli(["query", "cat"])

        assert mock_get.call_count == 2
        # First call should be /expand
        first_url = mock_get.call_args_list[0][0][0]
        assert "/expand" in first_url
        # Second call should be /search with expanded terms
        second_url = mock_get.call_args_list[1][0][0]
        assert "/search" in second_url
        second_params = mock_get.call_args_list[1][1].get("params", {})
        assert second_params.get("q") == "cat feline kitten"

    @patch("qmd_wrapper.requests.get")
    def test_query_mode_falls_back_on_expand_failure(self, mock_get, capsys):
        """If /expand fails, query mode falls back to plain /search with original query."""
        expand_resp = MagicMock()
        expand_resp.raise_for_status.side_effect = requests.RequestException("500")
        search_resp = MagicMock(status_code=200, json=lambda: {"results": _make_results(1)})
        search_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [expand_resp, search_resp]

        _run_cli(["query", "cat"])

        assert mock_get.call_count == 2
        # Search should use the original query text
        second_params = mock_get.call_args_list[1][1].get("params", {})
        assert second_params.get("q") == "cat"

    @patch("qmd_wrapper.requests.get")
    def test_search_mode_does_not_call_expand(self, mock_get, capsys):
        """search subcommand should NOT call /expand — only /search."""
        search_resp = MagicMock(status_code=200, json=lambda: {"results": []})
        search_resp.raise_for_status = MagicMock()
        mock_get.return_value = search_resp

        _run_cli(["search", "cat"])

        assert mock_get.call_count == 1
        assert "/search" in mock_get.call_args[0][0]
        assert "/expand" not in mock_get.call_args[0][0]

    @patch("qmd_wrapper.requests.get")
    def test_vsearch_mode_does_not_call_expand(self, mock_get, capsys):
        """vsearch subcommand should NOT call /expand — only /query."""
        query_resp = MagicMock(status_code=200, json=lambda: {"answer": "result"})
        query_resp.raise_for_status = MagicMock()
        mock_get.return_value = query_resp

        _run_cli(["vsearch", "cat"])

        assert mock_get.call_count == 1
        assert "/query" in mock_get.call_args[0][0]
        assert "/expand" not in mock_get.call_args[0][0]

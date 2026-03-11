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

        # Verify it called the same /query endpoint
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

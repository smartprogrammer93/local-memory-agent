#!/usr/bin/env python3
"""
QMD Wrapper — CLI interface for the Local Memory Agent.

Provides qmd-style subcommands (search, query, vsearch, get, status, etc.)
that talk to the running local memory agent HTTP API and format results as QMD snippets.
"""

import argparse
import os
import sys

import requests

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8888")
DEFAULT_RESULTS = int(os.getenv("MEMORY_RESULTS", "5"))
QMD_INDEX = os.path.expanduser(
    "~/.openclaw/agents/main/qmd/xdg-cache/qmd/index.sqlite"
)


def _get_qmd_anchor_docid():
    """Return a real document hash from QMD's SQLite index to use as anchor."""
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(QMD_INDEX)
        row = conn.execute(
            "SELECT hash FROM documents WHERE active=1 ORDER BY path LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return "memory"  # last-resort fallback (will likely be skipped by OpenClaw)


# ─── Helpers ───────────────────────────────────────────────────


def _query_agent(query_text, n=None, files_mode=False, min_score=None, json_output=False):
    """Search memories — uses fast /search endpoint (no LLM) for low-latency results.

    The full LLM-synthesis /query endpoint is available via the HTTP API directly
    when rich synthesized answers are needed, but is too slow for tool integration.
    """
    import json as _json
    n = n or DEFAULT_RESULTS
    url = f"{MEMORY_AGENT_URL}/search"

    try:
        resp = requests.get(url, params={"q": query_text}, timeout=120)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout) as exc:
        if json_output:
            print("[]")
        else:
            print(f"local-memory-agent-cli: agent unreachable ({exc})", file=sys.stderr)
        return
    except requests.RequestException as exc:
        if json_output:
            print("[]")
        else:
            print(f"local-memory-agent-cli: request failed ({exc})", file=sys.stderr)
        return

    data = resp.json()

    # The agent may return a simple answer string or structured results.
    results = data.get("results", [])

    # If the response is a flat answer (no structured results), wrap it.
    if not results and "answer" in data:
        results = [
            {
                "docid": "agent-answer",
                "filepath": "",
                "title": "Agent Answer",
                "score": 1.0,
                "snippet": data["answer"],
            }
        ]

    # Apply min-score filter
    if min_score is not None:
        threshold = min_score / 100.0
        results = [r for r in results if r.get("score", 0) >= threshold]

    # Limit to n results
    results = results[:n]

    if json_output:
        # Return JSON array in QMD format OpenClaw expects: [{docid, score, snippet}, ...]
        # docid must be a real hash from QMD's SQLite index — we use the MEMORY.md hash
        # as anchor while injecting our synthesized answer as the snippet.
        anchor_docid = _get_qmd_anchor_docid()
        output = [
            {
                "docid": anchor_docid,
                "score": r.get("score", 1.0),
                "snippet": r.get("snippet", r.get("title", "")),
            }
            for r in results
        ]
        print(_json.dumps(output))
    elif files_mode:
        for r in results:
            score_pct = int(r.get("score", 0) * 100)
            print(f"{r.get('docid', '')},{score_pct},{r.get('filepath', '')}")
    else:
        _print_qmd_snippets(results)


def _print_qmd_snippets(results):
    """Format results as QMD-style snippets."""
    for i, r in enumerate(results):
        if i > 0:
            print("---")
        docid = r.get("docid", "unknown")
        title = r.get("title", docid)
        score_pct = int(r.get("score", 0) * 100)
        snippet = r.get("snippet", "")
        filepath = r.get("filepath", "")

        uri = f"qmd://{filepath}" if filepath else f"qmd://{docid}"
        print(f"URI: {uri}")
        print(f"Title: {title}")
        print(f"Score: {score_pct}%")
        print(f"\n{snippet}")


def _read_file(filespec, lines=None, from_line=None):
    """Read a file from disk. filespec can be 'path' or 'path:line'."""
    parts = filespec.rsplit(":", 1)
    filepath = parts[0]
    start = from_line or 1

    if len(parts) == 2 and parts[1].isdigit():
        start = int(parts[1])

    if from_line is not None:
        start = from_line

    try:
        with open(filepath, "r") as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"local-memory-agent-cli: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"local-memory-agent-cli: permission denied: {filepath}", file=sys.stderr)
        sys.exit(1)

    # Convert to 0-based index
    start_idx = max(start - 1, 0)

    if lines is not None:
        selected = all_lines[start_idx : start_idx + lines]
    else:
        selected = all_lines[start_idx:]

    sys.stdout.write("".join(selected))


# ─── Subcommand handlers ──────────────────────────────────────


def cmd_search(args):
    _query_agent(
        " ".join(args.query),
        n=args.n,
        files_mode=args.files,
        min_score=args.min_score,
        json_output=getattr(args, "json", False),
    )


def cmd_get(args):
    _read_file(args.file, lines=args.l, from_line=getattr(args, "from"))


def cmd_multi_get(args):
    for filespec in args.files:
        _read_file(filespec, lines=args.l, from_line=getattr(args, "from"))


def cmd_status(args):
    url = f"{MEMORY_AGENT_URL}/status"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout) as exc:
        print(f"local-memory-agent-cli: agent unreachable ({exc})", file=sys.stderr)
        return
    except requests.RequestException as exc:
        print(f"local-memory-agent-cli: request failed ({exc})", file=sys.stderr)
        return

    data = resp.json()
    doc_count = data.get("total_memories", 0)
    last_updated = data.get("last_updated", "unknown")
    print("QMD Status")
    print("Collections: 1")
    print(f"Documents: {doc_count}")
    print(f"Last updated: {last_updated}")


def cmd_noop(args):
    print(f"local-memory-agent-cli: {args._subcmd} is a no-op (handled by local memory agent)")


def cmd_mcp(args):
    print("local-memory-agent-cli: mcp subcommand is not supported")
    sys.exit(1)


# ─── CLI ───────────────────────────────────────────────────────


def build_parser():
    parser = argparse.ArgumentParser(
        prog="qmd_wrapper",
        description="QMD-style CLI for the Local Memory Agent",
    )
    sub = parser.add_subparsers(dest="command")

    # search / query / vsearch — all do the same thing
    for name in ("search", "query", "vsearch"):
        p = sub.add_parser(name, help=f"Search memories ({name})")
        p.add_argument("query", nargs="+", help="Search query")
        p.add_argument("-n", type=int, default=DEFAULT_RESULTS, help="Max results")
        p.add_argument("--files", action="store_true", help="CSV output: docid,score,filepath")
        p.add_argument("--min-score", type=int, default=None, help="Min confidence %%")
        p.add_argument("--json", action="store_true", help="JSON output (ignored, always plain text)")
        p.add_argument("-c", "--collection", default=None, help="Collection name (ignored, searches all)")
        p.set_defaults(func=cmd_search)

    # get
    p = sub.add_parser("get", help="Read a file from disk")
    p.add_argument("file", help="File path (optional :line suffix)")
    p.add_argument("-l", type=int, default=None, help="Number of lines to read")
    p.add_argument("--from", type=int, default=None, dest="from", help="Start line")
    p.set_defaults(func=cmd_get)

    # multi-get
    p = sub.add_parser("multi-get", help="Read multiple files from disk")
    p.add_argument("files", nargs="+", help="File paths (optional :line suffix)")
    p.add_argument("-l", type=int, default=None, help="Number of lines to read")
    p.add_argument("--from", type=int, default=None, dest="from", help="Start line")
    p.set_defaults(func=cmd_multi_get)

    # status
    p = sub.add_parser("status", help="Agent status")
    p.set_defaults(func=cmd_status)

    # no-op subcommands
    for name in ("update", "embed", "collection", "ls", "context", "cleanup"):
        p = sub.add_parser(name, help=f"{name} (no-op)")
        p.set_defaults(func=cmd_noop, _subcmd=name)

    # mcp
    p = sub.add_parser("mcp", help="MCP subcommand (not supported)")
    p.set_defaults(func=cmd_mcp)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

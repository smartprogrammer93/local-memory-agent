"""
Agent Memory Layer — Always-On Local Memory Agent

A lightweight background agent that continuously processes, consolidates,
and serves memory via a local LLM over OpenAI-compatible API.

Usage:
    python agent.py                          # watch ./inbox, serve on :8888
    python agent.py --watch ./inbox --port 9000
    python agent.py --consolidate-every 15   # consolidate every 15 min
    python agent.py --watch-memory /path/to/memory  # also watch a memory folder

Query:
    curl "http://localhost:8888/query?q=what+do+you+know"
    curl -X POST http://localhost:8888/ingest -d '{"text": "some info"}'
"""

import argparse
import asyncio
import logging
import os
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import build_agents, MemoryOrchestrator
from media import prepare_file, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, PDF_EXTENSIONS, TEXT_EXTENSIONS
from tools import (
    init_db,
    read_all_memories,
    get_memory_stats,
    delete_memory,
    clear_all_memories,
    search_memories_fast,
)

load_dotenv()

# ─── Config ────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.8.188:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Local LLM-Q6_K.gguf")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
DB_PATH = os.getenv("MEMORY_DB", "memory.db")

ALL_SUPPORTED = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | PDF_EXTENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M]",
)
log = logging.getLogger("memory-agent")

# ─── MemoryAgent ──────────────────────────────────────────────


class MemoryAgent:
    def __init__(self):
        self.client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.agents = build_agents(self.client, LLM_MODEL)
        self.orchestrator = MemoryOrchestrator(self.agents, self.client, LLM_MODEL)
        init_db()

    async def run(self, message: str) -> str:
        """Route a free-form message to the appropriate agent."""
        return await self.orchestrator.route(message)

    async def run_multimodal(self, text: str, file_path: str) -> str:
        """Preprocess a file and send its content to the ingest agent."""
        content = await prepare_file(Path(file_path))
        if isinstance(content, list):
            # Multimodal content (images/video) — build message with text + media
            parts = [{"type": "text", "text": f"Remember this (source: {Path(file_path).name}): {text}"}]
            parts.extend(content)
            msg = [{"role": "user", "content": parts}]
            return await self.agents["ingest"].run(msg)
        else:
            # Text content — combine with user text
            combined = f"{text}\n\n{content}" if text else content
            return await self.agents["ingest"].run(
                f"Remember this information (source: {Path(file_path).name}):\n\n{combined}"
            )

    async def ingest(self, text: str, source: str = "") -> str:
        """Send text to the ingest agent with optional source prefix."""
        msg = (
            f"Remember this information (source: {source}):\n\n{text}"
            if source
            else f"Remember this information:\n\n{text}"
        )
        return await self.agents["ingest"].run(msg)

    async def ingest_file(self, file_path: str) -> str:
        """Preprocess a file and send to the ingest agent."""
        path = Path(file_path)
        content = await prepare_file(path)
        if isinstance(content, list):
            # Multimodal content — send as structured message
            msg = [{"role": "user", "content": content}]
            return await self.agents["ingest"].run(msg)
        else:
            return await self.agents["ingest"].run(
                f"Remember this information (source: {path.name}):\n\n{content}"
            )

    async def consolidate(self) -> str:
        """Trigger the consolidation agent."""
        return await self.agents["consolidate"].run(
            "Consolidate unconsolidated memories. Find connections and patterns."
        )

    async def query(self, question: str) -> str:
        """Send a question to the query agent."""
        return await self.agents["query"].run(
            f"Based on my memories, answer: {question}"
        )


# ─── File Watcher ──────────────────────────────────────────────


async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5, source_tag: str = ""):
    """Watch a folder for new or modified files and ingest them."""
    folder.mkdir(parents=True, exist_ok=True)
    db = init_db()
    # Add file_mtime column if missing (handles existing databases)
    try:
        db.execute("SELECT file_mtime FROM processed_files LIMIT 0")
    except sqlite3.OperationalError:
        db.execute("ALTER TABLE processed_files ADD COLUMN file_mtime REAL")
        db.commit()
    log.info(f"Watching: {folder}/")

    while True:
        try:
            for f in sorted(folder.iterdir()):
                if f.name.startswith("."):
                    continue
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue

                current_mtime = f.stat().st_mtime
                row = db.execute(
                    "SELECT file_mtime FROM processed_files WHERE path = ?", (str(f),)
                ).fetchone()

                if row:
                    stored_mtime = row["file_mtime"]
                    if stored_mtime is not None and current_mtime <= stored_mtime:
                        continue
                    # File was modified since last ingestion
                    log.info(f"Modified file: {f.name}")
                else:
                    log.info(f"New file: {f.name}")

                try:
                    if source_tag:
                        content = await prepare_file(f)
                        if isinstance(content, list):
                            await agent.ingest_file(str(f))
                        else:
                            await agent.ingest(content, source=source_tag)
                    else:
                        await agent.ingest_file(str(f))
                except Exception as file_err:
                    log.error(f"Error ingesting {f.name}: {file_err}")

                now = datetime.now(timezone.utc).isoformat()
                db.execute(
                    "INSERT OR REPLACE INTO processed_files (path, processed_at, file_mtime) VALUES (?, ?, ?)",
                    (str(f), now, current_mtime),
                )
                db.commit()
        except Exception as e:
            log.error(f"Watch error: {e}")

        await asyncio.sleep(poll_interval)


# ─── Consolidation Timer ──────────────────────────────────────


async def consolidation_loop(agent: MemoryAgent, interval_minutes: int = 30):
    """Run consolidation periodically."""
    log.info(f"Consolidation: every {interval_minutes} minutes")
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            db = init_db()
            count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            db.close()
            if count >= 2:
                log.info(f"Running consolidation ({count} unconsolidated memories)...")
                result = await agent.consolidate()
                log.info(f"Consolidation result: {result[:100]}")
            else:
                log.info(f"Skipping consolidation ({count} unconsolidated memories)")
        except Exception as e:
            log.error(f"Consolidation error: {e}")


# ─── HTTP API ──────────────────────────────────────────────────


def build_http(agent: MemoryAgent, watch_path: str = "./inbox"):
    app = web.Application()

    async def handle_query(request: web.Request):
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        answer = await agent.query(q)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        source = data.get("source", "api")
        result = await agent.ingest(text, source=source)
        return web.json_response({"status": "ingested", "response": result})

    async def handle_ingest_file(request: web.Request):
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            return web.json_response({"error": "missing 'file' field in multipart"}, status=400)

        filename = field.filename or "upload"
        inbox = Path(watch_path)
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / filename

        with open(dest, "wb") as fp:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                fp.write(chunk)

        try:
            result = await agent.ingest_file(str(dest))
            return web.json_response({"status": "ingested", "filename": filename, "response": result})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_consolidate(request: web.Request):
        result = await agent.consolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_status(request: web.Request):
        stats = get_memory_stats()
        return web.json_response(stats)

    async def handle_memories(request: web.Request):
        data = read_all_memories()
        return web.json_response(data)

    async def handle_delete(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        memory_id = data.get("memory_id")
        if not memory_id:
            return web.json_response({"error": "missing 'memory_id' field"}, status=400)
        result = delete_memory(int(memory_id))
        return web.json_response(result)

    async def handle_clear(request: web.Request):
        result = clear_all_memories(inbox_path=watch_path)
        return web.json_response(result)

    async def handle_health(request: web.Request):
        return web.json_response({"status": "ok"})

    async def handle_search(request: web.Request):
        """Fast keyword/FTS search over stored memories — no LLM, returns instantly."""
        q = request.rel_url.query.get("q", "").strip()
        n = int(request.rel_url.query.get("n", "5"))
        if not q:
            return web.json_response({"results": []})
        results = search_memories_fast(q, n)
        return web.json_response({"results": results})

    app.router.add_get("/search", handle_search)
    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/ingest-file", handle_ingest_file)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/memories", handle_memories)
    app.router.add_post("/delete", handle_delete)
    app.router.add_post("/clear", handle_clear)
    app.router.add_get("/health", handle_health)

    return app


# ─── Main ──────────────────────────────────────────────────────


async def main_async(args):
    agent = MemoryAgent()

    log.info("Agent Memory Layer starting")
    log.info(f"   Model: {LLM_MODEL}")
    log.info(f"   API: {LLM_BASE_URL}")
    log.info(f"   Database: {DB_PATH}")
    log.info(f"   Watch: {args.watch}")
    if args.watch_memory:
        log.info(f"   Watch memory: {args.watch_memory}")
    log.info(f"   Consolidate: every {args.consolidate_every}m")
    log.info(f"   HTTP: http://localhost:{args.port}")
    log.info("")

    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
    ]

    if args.watch_memory:
        tasks.append(asyncio.create_task(
            watch_folder(agent, Path(args.watch_memory), source_tag="openclaw-memory")
        ))

    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"Agent running on http://localhost:{args.port}")
    log.info("")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch for new files")
    parser.add_argument("--watch-memory", default=None, help="Additional folder to watch (e.g. OpenClaw memory)")
    parser.add_argument("--port", type=int, default=8888, help="HTTP API port")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Consolidation interval in minutes")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()

    def shutdown(sig):
        log.info(f"Shutting down (signal {sig})...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()
        log.info("Agent stopped.")


if __name__ == "__main__":
    main()

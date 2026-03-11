# Qwen Memory Agent

**An always-on AI memory agent powered by a local Qwen 3.5 model — no cloud APIs required.**

> Forked from [GoogleCloudPlatform/always-on-memory-agent](https://github.com/Shubhamsaboo/always-on-memory-agent), replacing Google ADK + Gemini with a local Qwen LLM over an OpenAI-compatible API.

Most AI agents have amnesia. They process information when asked, then forget everything. This project gives agents a persistent, evolving memory that runs 24/7 as a lightweight background process — continuously processing, consolidating, and connecting information.

No vector database. No embeddings. No cloud dependencies. Just a local LLM that reads, thinks, and writes structured memory.

## How It Works

Three specialized agents collaborate around a shared SQLite memory store:

| Agent | Role |
|---|---|
| **IngestAgent** | Extracts summary, entities, topics, and importance from incoming information |
| **ConsolidateAgent** | Periodically finds cross-connections and generates insights (like a brain during sleep) |
| **QueryAgent** | Answers questions by synthesizing memories with source citations |

**Supported file types (27 total):** text (`.txt`, `.md`, `.json`, `.csv`, `.log`, `.xml`, `.yaml`, `.yml`), images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.svg`), audio (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`), video (`.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`), and PDFs.

## Requirements

- **Python 3.10+**
- **llama-server** (or any OpenAI-compatible API server) running **Qwen 3.5** (tested with `Qwen3.5-9B-Q6_K.gguf`)
- **Whisper STT** binary — for audio transcription (e.g. [whisper.cpp](https://github.com/ggerganov/whisper.cpp))
- **ffmpeg** — for video frame extraction
- **SQLite 3** (bundled with Python)

## Quick Start

### 1. Install

```bash
git clone https://github.com/smartprogrammer93/qwen-memory-agent.git
cd qwen-memory-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your settings:
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `QWEN_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API endpoint |
| `QWEN_MODEL` | `Qwen3.5-9B-Q6_K.gguf` | Model name/path |
| `QWEN_API_KEY` | `none` | API key (use `none` for local servers) |
| `MEMORY_DB` | `memory.db` | SQLite database path |
| `CONSOLIDATE_EVERY` | `30` | Consolidation interval in minutes |
| `WATCH_DIR` | `./inbox` | Folder to watch for new files |
| `API_PORT` | `8888` | HTTP API port |
| `WHISPER_BIN` | `/usr/local/bin/whisper-stt` | Path to Whisper STT binary |
| `WHISPER_MODEL` | `turbo` | Whisper model variant |
| `IMAGE_MAX_PX` | `1024` | Max image dimension for vision API |
| `OPENCLAW_MEMORY_DIR` | *(empty)* | Optional: additional folder to watch |

### 3. Start your LLM server

```bash
# Example with llama-server:
llama-server -m Qwen3.5-9B-Q6_K.gguf --port 8080
```

### 4. Run the agent

```bash
python agent.py
```

The agent is now:
- Watching `./inbox/` for new files
- Consolidating memories every 30 minutes
- Serving queries at `http://localhost:8888`

### 5. Feed it information

```bash
# Drop any file into inbox/
echo "Important meeting notes" > inbox/notes.txt
cp report.pdf inbox/
cp recording.mp3 inbox/

# Or use the HTTP API
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "AI agents are the future", "source": "article"}'
```

### 6. Query

```bash
curl "http://localhost:8888/query?q=what+do+you+know"
```

### 7. Dashboard (optional)

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

## OpenClaw Memory Integration

Set `OPENCLAW_MEMORY_DIR` in `.env` to an OpenClaw workspace memory folder (e.g. `~/.openclaw/workspace/memory`). The agent will watch that directory for new/modified files and automatically ingest them, giving your OpenClaw agent persistent long-term memory.

You can also use the CLI flag:

```bash
python agent.py --watch-memory /path/to/openclaw/memory
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/status` | GET | Memory statistics (counts) |
| `/memories` | GET | List all stored memories |
| `/ingest` | POST | Ingest text (`{"text": "...", "source": "..."}`) |
| `/ingest-file` | POST | Upload file (multipart form) |
| `/query?q=...` | GET | Query memory with a question |
| `/consolidate` | POST | Trigger manual consolidation |
| `/delete` | POST | Delete a memory (`{"memory_id": 1}`) |
| `/clear` | POST | Delete all memories (full reset) |

## CLI Options

```bash
python agent.py [options]

  --watch DIR              Folder to watch for files (default: ./inbox)
  --watch-memory DIR       Additional memory folder to watch
  --port PORT              HTTP API port (default: 8888)
  --consolidate-every MIN  Consolidation interval (default: 30)
```

## Project Structure

```
qwen-memory-agent/
├── agent.py          # Main always-on agent (file watcher, HTTP server, consolidation loop)
├── agents.py         # Agent definitions & orchestrator routing
├── llm.py            # QwenAgent — tool-calling LLM agent over OpenAI-compatible API
├── tools.py          # SQLite memory database operations
├── media.py          # Media preprocessing (images, audio, video, PDFs)
├── dashboard.py      # Streamlit UI
├── deploy.sh         # Systemd deployment helper
├── systemd/          # Systemd service file
├── tests/            # Test suite
├── inbox/            # Drop files here for auto-ingestion
└── memory.db         # SQLite database (created automatically)
```

## Deployment

A systemd service file is included for running as a background daemon:

```bash
bash deploy.sh
```

## Why Local Qwen?

This agent runs continuously. Privacy, cost, and latency matter:

- **Private**: All data stays on your machine — no cloud calls
- **Free**: Zero API costs, runs on consumer hardware
- **Fast**: Low-latency with a local quantized model
- **Smart enough**: Extracts structure, finds connections, synthesizes answers

## Built With

- [Qwen 3.5](https://huggingface.co/Qwen) via OpenAI-compatible API
- SQLite for persistent memory storage
- aiohttp for the HTTP API
- Streamlit for the dashboard
- Whisper for audio transcription
- ffmpeg for video processing

## License

MIT

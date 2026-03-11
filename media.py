"""
Media preprocessing — image handling and file dispatch for OpenAI vision API.

Resizes images, encodes to base64, and routes files to the correct handler
based on extension.
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

log = logging.getLogger("memory-agent")

# ─── Extension groups ─────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}


# ─── Image preprocessing ─────────────────────────────────────


async def prepare_image(file_path: Path, max_px: int = 1024) -> list:
    """Open an image, resize so the long edge <= max_px, and return OpenAI vision content.

    Args:
        file_path: Path to the image file.
        max_px: Maximum pixel size for the long edge.

    Returns:
        OpenAI multimodal content list with text prompt and base64 image.
    """
    img = Image.open(file_path)

    # Convert palette/RGBA modes to RGB for JPEG output
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if long edge exceeds max_px
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_px:
        scale = max_px / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Encode to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    filename = file_path.name
    return [
        {"type": "text", "text": f"Analyze this image (source: {filename}):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]


# ─── Stub handlers for non-image media ───────────────────────


async def transcribe_audio(file_path: Path) -> str:
    """Transcribe audio using the Whisper STT binary.

    Shells out to WHISPER_BIN (env var, default /root/.local/bin/whisper-stt)
    and returns the transcript text.
    """
    whisper_bin = os.environ.get("WHISPER_BIN", "/root/.local/bin/whisper-stt")
    whisper_model = os.environ.get("WHISPER_MODEL", "")

    cmd = [whisper_bin]
    if whisper_model:
        cmd.extend(["--model", whisper_model])
    cmd.append(str(file_path))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except (FileNotFoundError, PermissionError) as exc:
        return f"[Whisper failed: {str(exc)[:200]}]"

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        return f"[Whisper failed: {err[:200]}]"

    return stdout.decode("utf-8", errors="replace").strip()


async def extract_video_frames(file_path: Path, fps: float = 1 / 30) -> list[Path]:
    """Extract frames from a video file using ffmpeg.

    Args:
        file_path: Path to the video file.
        fps: Frames per second to extract (default: 1 frame per 30s).

    Returns:
        Sorted list of extracted frame paths.
    """
    if not file_path.exists():
        log.warning("Video file not found: %s", file_path)
        return []

    tmp_dir = tempfile.mkdtemp(prefix="vidframes_")
    out_pattern = os.path.join(tmp_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-i", str(file_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        out_pattern,
        "-y", "-loglevel", "error",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
    except (FileNotFoundError, PermissionError) as exc:
        log.error("ffmpeg not available: %s", exc)
        return []

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        log.error("ffmpeg failed (rc=%d): %s", proc.returncode, err[:300])
        return []

    frames = sorted(Path(tmp_dir).glob("frame_*.jpg"))
    return frames


async def prepare_video(file_path: Path) -> str | list:
    """Extract frames from a video and return multimodal content for the vision API.

    Extracts up to 10 frames, runs prepare_image on each, and combines them
    into a single content list.
    """
    if not file_path.exists():
        return f"[Video file not found: {file_path.name}]"

    frames = await extract_video_frames(file_path)
    if not frames:
        return f"[Video file: {file_path.name} — frame extraction failed]"

    # Cap at 10 frames
    frames = frames[:10]

    content: list = [
        {"type": "text", "text": f"Video frames from {file_path.name} ({len(frames)} frames):"},
    ]
    for frame in frames:
        try:
            parts = await prepare_image(frame)
            # Skip the per-image text prefix, keep only the image_url part
            for part in parts:
                if part.get("type") == "image_url":
                    content.append(part)
        except Exception as exc:
            log.warning("Failed to process frame %s: %s", frame.name, exc)

    return content


async def prepare_pdf(file_path: Path) -> str | list:
    """Extract content from a PDF — text first, falling back to page images.

    Tries pdfplumber for text extraction (up to 20 pages). If text is too short
    (likely a scanned doc), falls back to PyMuPDF page rendering at 150 DPI
    (up to 10 pages).
    """
    if not file_path.exists():
        return f"[PDF file not found: {file_path.name}]"

    # --- Try text extraction with pdfplumber ---
    try:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:20]:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        full_text = "\n\n".join(text_parts).strip()

        if len(full_text) > 100:
            return full_text

    except ImportError:
        log.debug("pdfplumber not installed, skipping text extraction")
    except Exception as exc:
        log.warning("pdfplumber failed on %s: %s", file_path.name, exc)

    # --- Fall back to image rendering with PyMuPDF ---
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        page_count = min(len(doc), 10)
        content: list = [
            {"type": "text", "text": f"PDF pages from {file_path.name} ({page_count} pages):"},
        ]

        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        doc.close()
        return content

    except ImportError:
        log.error("Neither pdfplumber nor PyMuPDF available for PDF processing")
        return f"[PDF file: {file_path.name} — no PDF library available]"
    except Exception as exc:
        log.error("PyMuPDF failed on %s: %s", file_path.name, exc)
        return f"[PDF file: {file_path.name} — rendering failed: {str(exc)[:200]}]"


# ─── File dispatcher ──────────────────────────────────────────


async def prepare_file(file_path: Path) -> str | list:
    """Route a file to the correct handler based on its extension.

    Args:
        file_path: Path to the file to process.

    Returns:
        OpenAI vision content list for images, or a string for other types.
    """
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return await prepare_image(file_path)

    if ext in AUDIO_EXTENSIONS:
        return await transcribe_audio(file_path)

    if ext in VIDEO_EXTENSIONS:
        return await prepare_video(file_path)

    if ext in PDF_EXTENSIONS:
        return await prepare_pdf(file_path)

    if ext in TEXT_EXTENSIONS:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return text

    return f"[Unsupported file type: {ext}]"

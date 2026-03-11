"""Tests for the media module — image, PDF, audio preprocessing and file dispatch."""
import asyncio
import base64
import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from PIL import Image

from media import (
    prepare_image,
    prepare_file,
    prepare_pdf,
    transcribe_audio,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
)


# ─── Helpers ──────────────────────────────────────────────────


def _make_test_image(width=200, height=100, suffix=".jpg", mode="RGB"):
    """Create a temporary test image file."""
    img = Image.new(mode, (width, height), color="red" if mode == "RGB" else (255, 0, 0, 128))
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fmt = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
    img.save(tmp.name, format=fmt)
    tmp.close()
    return Path(tmp.name)


def _make_text_pdf(text="Hello from PDF test fixture"):
    """Create a minimal PDF with text content using pdfplumber-compatible format."""
    import fitz

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=14)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── prepare_image ────────────────────────────────────────────


class TestPrepareImage:
    def test_returns_content_list_with_text_and_image(self):
        path = _make_test_image()
        result = _run(prepare_image(path))
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert path.name in result[0]["text"]
        path.unlink()

    def test_base64_data_is_valid_jpeg(self):
        path = _make_test_image()
        result = _run(prepare_image(path))
        url = result[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        b64_data = url.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        # JPEG files start with FF D8
        assert raw[:2] == b"\xff\xd8"
        # Verify PIL can open the decoded bytes
        img = Image.open(io.BytesIO(raw))
        assert img.format == "JPEG"
        path.unlink()

    def test_resize_large_image(self):
        path = _make_test_image(width=3000, height=2000)
        result = _run(prepare_image(path, max_px=512))
        # Decode and check dimensions
        b64_data = result[1]["image_url"]["url"].split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        assert max(img.size) <= 512
        # Long edge should be exactly 512
        assert max(img.size) == 512
        # Aspect ratio preserved: 3000:2000 = 3:2 → 512:341
        assert img.size == (512, 341)
        path.unlink()

    def test_small_image_not_resized(self):
        path = _make_test_image(width=200, height=100)
        result = _run(prepare_image(path, max_px=1024))
        b64_data = result[1]["image_url"]["url"].split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        assert img.size == (200, 100)
        path.unlink()

    def test_handles_rgba_mode(self):
        path = _make_test_image(width=100, height=100, suffix=".png", mode="RGBA")
        result = _run(prepare_image(path))
        assert len(result) == 2
        assert result[1]["type"] == "image_url"
        path.unlink()


# ─── prepare_pdf (text path) ─────────────────────────────────


class TestPreparePdf:
    def test_text_extraction_returns_string(self):
        path = _make_text_pdf("This is a test PDF with enough text to pass the length threshold. " * 5)
        result = _run(prepare_pdf(path))
        assert isinstance(result, str)
        assert "test PDF" in result
        path.unlink()

    def test_text_extraction_contains_content(self):
        long_text = "Important document content for extraction verification. " * 10
        path = _make_text_pdf(long_text)
        result = _run(prepare_pdf(path))
        assert isinstance(result, str)
        assert len(result) > 100
        assert "Important document" in result
        path.unlink()


# ─── prepare_file dispatcher ─────────────────────────────────


class TestPrepareFile:
    def test_jpg_routes_to_prepare_image(self):
        path = _make_test_image(suffix=".jpg")
        result = _run(prepare_file(path))
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        path.unlink()

    def test_png_routes_to_prepare_image(self):
        path = _make_test_image(suffix=".png", mode="RGB")
        result = _run(prepare_file(path))
        assert isinstance(result, list)
        assert result[1]["type"] == "image_url"
        path.unlink()

    def test_pdf_routes_to_prepare_pdf(self):
        path = _make_text_pdf("Dispatched PDF content for routing test. " * 5)
        result = _run(prepare_file(path))
        # Text PDF returns a string
        assert isinstance(result, str)
        assert "Dispatched PDF" in result
        path.unlink()

    def test_txt_reads_file_as_plain_text(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        tmp.write("Hello world")
        tmp.close()
        path = Path(tmp.name)
        result = _run(prepare_file(path))
        assert result == "Hello world"
        path.unlink()

    def test_json_reads_file_as_text(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write('{"key": "value"}')
        tmp.close()
        path = Path(tmp.name)
        result = _run(prepare_file(path))
        assert result == '{"key": "value"}'
        path.unlink()

    def test_unsupported_extension_returns_error(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        tmp.close()
        path = Path(tmp.name)
        result = _run(prepare_file(path))
        assert "[Unsupported file type: .xyz]" == result
        path.unlink()


# ─── transcribe_audio ────────────────────────────────────────


class TestTranscribeAudio:
    def test_calls_whisper_with_correct_args(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Hello from whisper", b"")
        mock_proc.returncode = 0

        with patch("media.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
             patch.dict("os.environ", {"WHISPER_BIN": "/usr/bin/whisper", "WHISPER_MODEL": ""}):
            result = _run(transcribe_audio(Path("/tmp/test.mp3")))

        assert result == "Hello from whisper"
        mock_exec.assert_called_once_with(
            "/usr/bin/whisper", "/tmp/test.mp3",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def test_passes_model_flag_when_set(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"transcript", b"")
        mock_proc.returncode = 0

        with patch("media.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
             patch.dict("os.environ", {"WHISPER_BIN": "/usr/bin/whisper", "WHISPER_MODEL": "large-v3"}):
            result = _run(transcribe_audio(Path("/tmp/test.wav")))

        assert result == "transcript"
        mock_exec.assert_called_once_with(
            "/usr/bin/whisper", "--model", "large-v3", "/tmp/test.wav",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def test_failure_returns_error_string(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"whisper: model not found")
        mock_proc.returncode = 1

        with patch("media.asyncio.create_subprocess_exec", return_value=mock_proc), \
             patch.dict("os.environ", {"WHISPER_BIN": "/usr/bin/whisper", "WHISPER_MODEL": ""}):
            result = _run(transcribe_audio(Path("/tmp/test.mp3")))

        assert "[Whisper failed:" in result
        assert "model not found" in result

    def test_binary_not_found_returns_error(self):
        with patch("media.asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No such file")), \
             patch.dict("os.environ", {"WHISPER_BIN": "/nonexistent/whisper", "WHISPER_MODEL": ""}):
            result = _run(transcribe_audio(Path("/tmp/test.mp3")))

        assert "[Whisper failed:" in result
        assert "No such file" in result

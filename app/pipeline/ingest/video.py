"""Video demo: transcribe the audio track (optional). Frames are not analysed."""

import logging
import shutil
import subprocess
from pathlib import Path

import httpx

from app.config import Settings

log = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 24 * 1024 * 1024  # stay under common 25 MB transcription limits


def process_video(path: Path, dest_dir: Path, settings: Settings, transport: httpx.BaseTransport | None = None) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"size_bytes": path.stat().st_size, "transcribed": False}
    if not (settings.transcription_base_url and settings.transcription_api_key):
        info["note"] = "transcription is not configured on this deployment"
        return info
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        info["note"] = "ffmpeg is not installed on this worker"
        return info
    audio = dest_dir / "_audio.mp3"
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)],
            check=True, capture_output=True, timeout=900,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        info["note"] = f"could not extract audio: {type(exc).__name__}"
        return info
    if audio.stat().st_size > MAX_AUDIO_BYTES:
        info["note"] = "audio track too long for the transcription service"
        audio.unlink(missing_ok=True)
        return info
    try:
        with httpx.Client(transport=transport, timeout=300) as client, audio.open("rb") as f:
            resp = client.post(
                settings.transcription_base_url.rstrip("/") + "/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.transcription_api_key}"},
                data={"model": settings.transcription_model, "response_format": "json"},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
            )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        info["note"] = f"transcription failed: {type(exc).__name__}"
        return info
    finally:
        audio.unlink(missing_ok=True)
    (dest_dir / "transcript.txt").write_text(text, encoding="utf-8")
    info.update(transcribed=True, transcript_chars=len(text))
    return info

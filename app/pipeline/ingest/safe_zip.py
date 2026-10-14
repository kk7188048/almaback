"""Hardened ZIP extraction: zip-slip, symlinks, zip bombs, nested archives, encrypted entries."""

import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from app.pipeline.errors import CorruptArchive, UnsafeArchive

NESTED_ARCHIVE_EXT = {".zip", ".jar", ".war", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
CHUNK = 64 * 1024


@dataclass
class ZipLimits:
    max_total_bytes: int
    max_entry_bytes: int
    max_ratio: int
    max_entries: int


@dataclass
class ExtractReport:
    files: int = 0
    bytes_written: int = 0
    skipped: list[str] = field(default_factory=list)


def _clean_name(name: str) -> PurePosixPath:
    if "\x00" in name:
        raise UnsafeArchive("archive entry contains a NUL byte")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise UnsafeArchive(f"archive entry has an absolute path: {name!r}")
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise UnsafeArchive(f"archive entry escapes the extraction directory: {name!r}")
    return path


def safe_extract(zip_path: Path, dest: Path, limits: ZipLimits) -> ExtractReport:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    report = ExtractReport()
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise CorruptArchive("the ZIP file is corrupt or not a ZIP archive") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > limits.max_entries:
            raise UnsafeArchive(f"archive has too many entries ({len(infos)} > {limits.max_entries})")
        # Declared sizes are attacker-controlled, so this is only an early reject; real bytes are counted below.
        if sum(i.file_size for i in infos) > limits.max_total_bytes:
            raise UnsafeArchive("archive would expand beyond the allowed size")

        for info in infos:
            rel = _clean_name(info.filename)
            if not rel.parts:
                continue
            if info.flag_bits & 0x1:
                raise UnsafeArchive("encrypted archives are not supported")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                report.skipped.append(f"{rel} (symlink)")
                continue
            target = (root / Path(*rel.parts)).resolve()
            if root != target and root not in target.parents:
                raise UnsafeArchive(f"archive entry escapes the extraction directory: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if Path(rel.name).suffix.lower() in NESTED_ARCHIVE_EXT:
                report.skipped.append(f"{rel} (nested archive)")
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            try:
                with zf.open(info) as src, target.open("wb") as out:
                    while chunk := src.read(CHUNK):
                        written += len(chunk)
                        if written > limits.max_entry_bytes:
                            raise UnsafeArchive(f"entry {rel} exceeds the per-file size limit")
                        if report.bytes_written + written > limits.max_total_bytes:
                            raise UnsafeArchive("archive expanded beyond the allowed size")
                        if written > 1024 * 1024 and written / max(info.compress_size, 1) > limits.max_ratio:
                            raise UnsafeArchive(f"entry {rel} has a suspicious compression ratio (zip bomb)")
                        out.write(chunk)
            except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, NotImplementedError) as exc:
                raise CorruptArchive(f"could not read entry {rel}: {exc}") from exc
            report.bytes_written += written
            report.files += 1
    return report


def project_root(extracted: Path) -> Path:
    """If the archive is a single top-level folder (the usual case), descend into it."""
    entries = [p for p in extracted.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted

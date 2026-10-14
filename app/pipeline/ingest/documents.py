"""Turn uploaded PDFs / documentation files into plain text files the context builder can read."""

import re
import zipfile
from pathlib import Path

from app.pipeline.errors import UnsupportedFile

MAX_PDF_PAGES = 300
MAX_TEXT_CHARS = 600_000
MAX_DOCX_XML_BYTES = 40 * 1024 * 1024
TEXT_EXT = {".md", ".markdown", ".txt", ".rst", ".adoc", ".csv", ".json", ".yaml", ".yml"}


def extract_pdf_text(path: Path) -> tuple[str, list[str]]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    warnings: list[str] = []
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnsupportedFile("password-protected PDFs are not supported", code="encrypted_pdf")
        pages = reader.pages
        n = len(pages)
        if n > MAX_PDF_PAGES:
            warnings.append(f"PDF has {n} pages; only the first {MAX_PDF_PAGES} were read")
        chunks: list[str] = []
        size = 0
        for i in range(min(n, MAX_PDF_PAGES)):
            text = pages[i].extract_text() or ""
            chunks.append(f"\n\n--- page {i + 1} ---\n{text}")
            size += len(text)
            if size > MAX_TEXT_CHARS:
                warnings.append("PDF text truncated")
                break
    except UnsupportedFile:
        raise
    except (PyPdfError, ValueError, KeyError, RecursionError, OSError) as exc:
        raise UnsupportedFile(f"the PDF could not be read: {exc}", code="unreadable_pdf") from exc
    text = "".join(chunks).strip()
    if not text:
        warnings.append("PDF contains no extractable text (scanned images are not OCR'd)")
    return text, warnings


def extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            info = zf.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise UnsupportedFile("DOCX document is too large", code="docx_too_large")
            with zf.open(info) as f:
                xml = f.read(MAX_DOCX_XML_BYTES + 1)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise UnsupportedFile("the DOCX file could not be read", code="unreadable_docx") from exc
    if len(xml) > MAX_DOCX_XML_BYTES:
        raise UnsupportedFile("DOCX document is too large", code="docx_too_large")
    text = re.sub(r"</w:p>", "\n", xml.decode("utf-8", "replace"))
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_TEXT_CHARS]


def convert_document(path: Path, dest_dir: Path, display_name: str) -> list[str]:
    """Write a plain-text rendition of `path` into dest_dir. Returns warnings."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(display_name).suffix.lower()
    stem = Path(display_name).stem
    head = path.read_bytes()[:1024] if path.exists() else b""
    warnings: list[str] = []
    if head.lstrip().startswith(b"%PDF-") or ext == ".pdf":
        text, warnings = extract_pdf_text(path)
        (dest_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    elif ext == ".docx":
        (dest_dir / f"{stem}.txt").write_text(extract_docx_text(path), encoding="utf-8")
    elif ext in TEXT_EXT:
        data = path.read_bytes()[: MAX_TEXT_CHARS * 2].decode("utf-8", "replace")
        (dest_dir / f"{stem}{ext}").write_text(data[:MAX_TEXT_CHARS], encoding="utf-8")
    else:
        raise UnsupportedFile(f"unsupported documentation format: {ext or 'unknown'}", code="unsupported_doc")
    return warnings

"""Content sniffing. Never trust file extensions or client-declared content types."""

ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
EXEC_MAGICS = (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


def is_zip(head: bytes) -> bool:
    return head.startswith(ZIP_MAGICS)


def is_pdf(head: bytes) -> bool:
    return head[:1024].lstrip(b"\x00\r\n\t ").startswith(b"%PDF-")


def is_video(head: bytes) -> bool:
    if len(head) >= 12 and head[4:8] == b"ftyp":  # mp4 / mov / m4v
        return True
    if head.startswith(b"\x1a\x45\xdf\xa3"):  # matroska / webm
        return True
    return head.startswith(b"RIFF") and head[8:12] == b"AVI "


def is_executable(head: bytes) -> bool:
    return head.startswith(EXEC_MAGICS)


def looks_like_text(head: bytes) -> bool:
    if not head:
        return True
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # a multi-byte char cut at the buffer edge is fine; anything earlier is binary
        if exc.start < len(head) - 4:
            return False
    return True


def validate_upload(kind: str, head: bytes) -> str | None:
    """Return an error string if `head` does not look like the declared upload kind."""
    if is_executable(head):
        return "executable files are not accepted"
    if kind == "zip" and not is_zip(head):
        return "file is not a ZIP archive"
    if kind == "pdf" and not is_pdf(head):
        return "file is not a PDF"
    if kind == "video" and not is_video(head):
        return "file is not a supported video (mp4/mov/webm/mkv/avi)"
    if kind == "docs" and not (is_pdf(head) or is_zip(head) or looks_like_text(head)):
        return "documentation must be text, markdown, PDF or DOCX"
    return None

"""
extractors.py

Given a filename (used only to detect type by extension) and raw bytes,
return the best-effort plaintext content of that file:

  * plaintext-ish files -> decoded text
  * images              -> OCR text via pytesseract (empty if none found)
  * PDFs                -> embedded text; if none, OCR each page; else empty
  * unknown extensions  -> sniff content; decode if it looks like text,
                           otherwise treat as binary and skip
"""

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Union
from PIL import Image
from pillow_heif import register_heif_opener

PLAINTEXT_EXTS = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".log",
    ".json", ".xml", ".html", ".htm", ".yaml", ".yml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".cs", ".go",
    ".rb", ".sh", ".bat", ".sql",
}


register_heif_opener()
IMAGE_EXTS = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}

def extract_text(filename: str, data: Union[bytes, Path, BinaryIO]) -> str:
    """Dispatch to the right extractor based on file extension.
    Never raises -- returns "" on any failure so a single bad file
    can't abort the whole cache build."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in PLAINTEXT_EXTS:
            return _extract_plaintext(data)
        elif ext in IMAGE_EXTS:
            return _extract_image(data)
        elif ext == ".pdf":
            return _extract_pdf(data)
        else:
            # Unknown/missing extension: sniff the content instead of
            # guessing blind. If it looks like text, treat it as plaintext;
            # if it looks binary, don't try to decode it.
            raw_bytes = _read_bytes(data)
            if is_probably_text(raw_bytes):
                return _decode_bytes(raw_bytes)
            return ""
    except Exception:
        return ""
    return ""


def is_probably_text(raw_bytes: bytes, sample_size: int = 8000) -> bool:
    """Heuristically decide whether raw_bytes looks like plaintext or binary.

    Mirrors the approach used by tools like `file`/git:
      1. A null byte is a strong signal of binary content.
      2. Otherwise, sample the start of the content and check what fraction
         of bytes fall outside the "text-like" range (printable ASCII,
         common whitespace/control chars). A high fraction -> binary.
    """
    if not raw_bytes:
        return True  # empty file -- nothing to lose by treating as text

    sample = raw_bytes[:sample_size]

    if b"\x00" in sample:
        return False

    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    nontext = sample.translate(None, text_chars)
    return (len(nontext) / len(sample)) <= 0.30


def _read_bytes(data: Union[bytes, Path, BinaryIO]) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, Path):
        return data.read_bytes()
    data.seek(0)
    return data.read()


def _decode_bytes(raw_bytes: bytes) -> str:
    """Decode raw bytes to text, trying common encodings in order."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def _extract_plaintext(data: Union[bytes, Path, BinaryIO]) -> str:
    raw_bytes = _read_bytes(data)
    return _decode_bytes(raw_bytes)


def _extract_image(data: Union[bytes, Path, BinaryIO]) -> str:
    import pytesseract

    if isinstance(data, bytes):
        img = Image.open(io.BytesIO(data))
    else:
        if isinstance(data, Path):
            img = Image.open(str(data))
        else:
            data.seek(0)
            img = Image.open(data)

    text = pytesseract.image_to_string(img)
    return text.strip()


def _extract_pdf(data: Union[bytes, Path, BinaryIO]) -> str:
    pdf_path, cleanup = _ensure_pdf_path(data)
    try:
        text = _pdf_embedded_text(pdf_path)
        if text.strip():
            return text
        return _pdf_ocr(pdf_path)
    finally:
        if cleanup:
            try:
                pdf_path.unlink()
            except OSError:
                pass


def _ensure_pdf_path(data: Union[bytes, Path, BinaryIO]) -> tuple[Path, bool]:
    if isinstance(data, Path):
        return data, False

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        if isinstance(data, bytes):
            tmp.write(data)
        else:
            data.seek(0)
            shutil.copyfileobj(data, tmp)
    finally:
        tmp.close()
    return Path(tmp.name), True


def _pdf_embedded_text(pdf_path: Path) -> str:
    import pdfplumber

    pieces = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pieces.append(t)
    return "\n".join(pieces)


def _pdf_ocr(pdf_path: Path) -> str:
    # Requires the `poppler` system package (for pdf2image) and `tesseract`.
    from pdf2image import convert_from_path
    import pytesseract

    pieces = []
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            num_pages = len(pdf.pages)
    except Exception:
        return ""

    for page_number in range(1, num_pages + 1):
        try:
            imgs = convert_from_path(
                str(pdf_path),
                first_page=page_number,
                last_page=page_number,
            )
        except Exception:
            continue
        if not imgs:
            continue
        try:
            pieces.append(pytesseract.image_to_string(imgs[0]))
        finally:
            imgs[0].close()
    return "\n".join(pieces).strip()

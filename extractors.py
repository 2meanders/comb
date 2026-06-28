"""
extractors.py

Given a filename (used only to detect type by extension) and raw bytes,
return the best-effort plaintext content of that file:

  * plaintext-ish files -> decoded text
  * images              -> OCR text via pytesseract (empty if none found)
  * PDFs                -> embedded text; if none, OCR each page; else empty
  * Office docs         -> .docx / .xlsx / .xlsm / .pptx text extraction
  * audio               -> transcription via OpenAI Whisper
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

# Modern, XML-based Office formats only. Legacy binary formats (.doc, .xls,
# .ppt -- pre-2007) use a completely different file structure and are NOT
# handled by python-docx/openpyxl/python-pptx. They'd fall through to the
# "unknown extension" sniffing path below and almost certainly be skipped
# as binary.
DOCX_EXTS = {".docx"}
XLSX_EXTS = {".xlsx", ".xlsm"}
PPTX_EXTS = {".pptx"}

# Audio formats transcribed via OpenAI Whisper. ffmpeg must be installed
# and on PATH -- whisper shells out to it to decode/resample audio.
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}

# Whisper model size used for transcription. "base" is a reasonable
# speed/accuracy default; swap for "tiny"/"small"/"medium"/"large" as needed.
WHISPER_MODEL_SIZE = "base"

# Loaded lazily on first use and cached for the lifetime of the process,
# since loading the model is expensive (downloads + GPU/CPU init) and we
# don't want to repeat that for every audio file in a batch.
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


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
        elif ext in DOCX_EXTS:
            return _extract_docx(data)
        elif ext in XLSX_EXTS:
            return _extract_xlsx(data)
        elif ext in PPTX_EXTS:
            return _extract_pptx(data)
        elif ext in AUDIO_EXTS:
            return _extract_audio(filename, data)
        else:
            # Unknown/missing extension: sniff the content instead of
            # guessing blind. If it looks like text, treat it as plaintext;
            # if it looks binary, don't try to decode it.
            raw_bytes = _read_bytes(data)
            if _is_probably_text(raw_bytes):
                return _decode_bytes(raw_bytes)
            return ""
    except Exception:
        return ""
    return ""


def _is_probably_text(raw_bytes: bytes, sample_size: int = 8000) -> bool:
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


def _as_file_like(data: Union[bytes, Path, BinaryIO]):
    """Return something openable by libraries that accept a path-or-file-obj
    (python-docx, openpyxl, python-pptx all accept a filename or a
    file-like object with .read())."""
    if isinstance(data, bytes):
        return io.BytesIO(data)
    if isinstance(data, Path):
        return str(data)
    data.seek(0)
    return data


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


def _extract_docx(data: Union[bytes, Path, BinaryIO]) -> str:
    """Extract text from a .docx: paragraphs, tables, and headers/footers."""
    import docx

    doc = docx.Document(_as_file_like(data))
    pieces = []

    for para in doc.paragraphs:
        if para.text:
            pieces.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text for cell in row.cells if cell.text]
            if cells:
                pieces.append("\t".join(cells))

    for section in doc.sections:
        for part in (section.header, section.footer):
            for para in part.paragraphs:
                if para.text:
                    pieces.append(para.text)

    return "\n".join(pieces)


def _extract_xlsx(data: Union[bytes, Path, BinaryIO]) -> str:
    """Extract text from .xlsx/.xlsm: every cell's value, sheet by sheet."""
    import openpyxl

    wb = openpyxl.load_workbook(_as_file_like(data), data_only=True, read_only=True)
    pieces = []
    try:
        for sheet in wb.worksheets:
            sheet_lines = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(v) for v in row if v is not None]
                if values:
                    sheet_lines.append("\t".join(values))
            if sheet_lines:
                pieces.append(f"# {sheet.title}")
                pieces.extend(sheet_lines)
    finally:
        wb.close()

    return "\n".join(pieces)


def _extract_pptx(data: Union[bytes, Path, BinaryIO]) -> str:
    """Extract text from .pptx: all text frames and table cells, per slide,
    plus speaker notes."""
    from pptx import Presentation

    prs = Presentation(_as_file_like(data))
    pieces = []

    for slide_number, slide in enumerate(prs.slides, start=1):
        slide_lines = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs)
                    if text:
                        slide_lines.append(text)
            elif shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text for cell in row.cells if cell.text]
                    if cells:
                        slide_lines.append("\t".join(cells))

        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text
            if notes_text:
                slide_lines.append(f"[notes] {notes_text}")

        if slide_lines:
            pieces.append(f"# Slide {slide_number}")
            pieces.extend(slide_lines)

    return "\n".join(pieces)


def _extract_audio(filename: str, data: Union[bytes, Path, BinaryIO]) -> str:
    """Transcribe an audio file to text using OpenAI Whisper.

    Whisper's `transcribe()` shells out to ffmpeg internally and expects a
    real file path, so bytes/file-like input is written to a temp file
    first (same pattern as PDF handling below).
    """
    audio_path, cleanup = _ensure_audio_path(filename, data)
    try:
        model = _get_whisper_model()
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Warning: failed to transcribe audio file {filename}: {e}")
    finally:
        if cleanup:
            try:
                audio_path.unlink()
            except OSError:
                pass


def _ensure_audio_path(filename: str, data: Union[bytes, Path, BinaryIO]) -> tuple[Path, bool]:
    if isinstance(data, Path):
        return data, False

    # Preserve the original suffix so ffmpeg's format sniffing has a hint
    # to work with (it also inspects file contents, but this avoids
    # ambiguity for less common containers).
    suffix = os.path.splitext(filename)[1] or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if isinstance(data, bytes):
            tmp.write(data)
        else:
            data.seek(0)
            shutil.copyfileobj(data, tmp)
    finally:
        tmp.close()
    return Path(tmp.name), True


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

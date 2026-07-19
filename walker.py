"""
walker.py

Walks a folder on disk and yields one Entry per "logical" file found,
descending into:
  * archives (zip, tar, and optionally 7z/rar -- see ARCHIVE_FORMATS)
  * .eml attachments (see CONTAINER_FORMATS)
including any of the above nested inside each other, in any combination.

Adding a new archive format is writing one ArchiveFormat class and
registering it; the recursive walk/nesting logic itself never changes.

A file that lives inside an archive or email gets a virtual_path like:
    archive.zip:inner/folder/file.txt
    message.eml:invoice.pdf
or, for nested archives/containers:
    outer.zip:inner.zip:deep/file.txt
    message.eml:attachments.zip:report.docx
"""

import fnmatch
import io
import shutil
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, BinaryIO, Optional, Union

CACHE_FILENAME = ".comb_cache.db"


def _is_comb_file(path: Path) -> bool:
    return path.is_file() and str(path).startswith(".comb_cache")


# Directory names that should never be descended into (checked against any
# path component, both on disk and inside archives).
IGNORED_DIR_NAMES = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".venv", "venv", "env", "dist", "build", ".next", ".cache",
}

# Filename glob patterns to skip, regardless of which directory they're in.
IGNORED_FILE_PATTERNS = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.lock",
    "*.log",
    ".DS_Store",
)


def _is_ignored_path(rel_path: str) -> bool:
    """Check a '/'-separated relative path (disk-relative or archive-internal)
    against the ignore rules."""
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]

    for part in parts[:-1]:
        if part in IGNORED_DIR_NAMES:
            return True

    for pattern in IGNORED_FILE_PATTERNS:
        if fnmatch.fnmatch(filename.lower(), pattern):
            return True

    return False


@dataclass
class Entry:
    virtual_path: str          # path used as the index key / display path
    mtime: float                # modification time, used to detect changes
    size: int                   # size in bytes, used to detect changes
    data_func: Callable[[], Union[bytes, Path, BinaryIO]]


# ── archive formats ─────────────────────────────────────────────────────────
#
# An ArchiveFormat knows how to list and lazily open the members of one kind
# of archive. `open_member` must be safe to call much later than
# `list_members` (extraction happens lazily, possibly from a worker thread),
# so it always reopens `source` fresh rather than reusing a handle.
#
# To support a new archive type (tar, 7z, ...), write one class implementing
# this interface and add an instance to ARCHIVE_FORMATS -- the recursive
# walk/nesting logic below is generic and doesn't need to change.

class ArchiveFormat(ABC):
    suffixes: tuple[str, ...] = ()

    @abstractmethod
    def list_members(self, source: Union[Path, BinaryIO]) -> list[tuple[str, int]]:
        """Return (name, size) for every regular file directly in the archive."""

    @abstractmethod
    def open_member(self, source: Union[Path, BinaryIO], name: str) -> BinaryIO:
        """Open one member for reading, reopening `source` from scratch."""


class _ClosingReader(io.BufferedReader):
    """Wraps a member stream so that closing it also closes the archive
    handle that was opened to produce it."""

    def __init__(self, archive, fileobj):
        super().__init__(fileobj)
        self._archive = archive

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._archive.close()


class ZipArchive(ArchiveFormat):
    suffixes = (".zip",)

    def _open(self, source: Union[Path, BinaryIO]) -> zipfile.ZipFile:
        if isinstance(source, Path):
            return zipfile.ZipFile(source, "r")
        source.seek(0)
        return zipfile.ZipFile(source, "r")

    def list_members(self, source):
        with self._open(source) as zf:
            return [(i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]

    def open_member(self, source, name):
        zf = self._open(source)
        return _ClosingReader(zf, zf.open(name))


class TarArchive(ArchiveFormat):
    # "r:*" auto-detects gzip/bz2/xz/no compression, so one class covers
    # all of these; the suffixes are just listed for clarity/matching.
    suffixes = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")

    def _open(self, source: Union[Path, BinaryIO]) -> tarfile.TarFile:
        if isinstance(source, Path):
            return tarfile.open(source, "r:*")
        source.seek(0)
        return tarfile.open(fileobj=source, mode="r:*")

    def list_members(self, source):
        with self._open(source) as tf:
            return [(m.name, m.size) for m in tf.getmembers() if m.isfile()]

    def open_member(self, source, name):
        tf = self._open(source)
        member_stream = tf.extractfile(name)
        if member_stream is None:
            # Shouldn't happen for a regular file (isfile() was already
            # checked in list_members), but guards against a malformed
            # archive or a symlink/device slipping through.
            tf.close()
            raise ValueError(f"tar member is not a readable regular file: {name}")
        return _ClosingReader(tf, member_stream)


# Optional formats that need a third-party library are only registered if
# that library is importable, so comb.py still works without every
# optional dependency installed -- an archive of that type just won't be
# descended into (it'll be treated as an unknown/binary file instead).
_OPTIONAL_FORMATS: list[ArchiveFormat] = []

try:
    import py7zr

    class SevenZipArchive(ArchiveFormat):
        suffixes = (".7z",)

        def list_members(self, source):
            if not isinstance(source, Path):
                source.seek(0)
            with py7zr.SevenZipFile(source, "r") as zf:
                return [
                    (info.filename, info.uncompressed)
                    for info in zf.list()
                    if not info.is_directory
                ]

        def open_member(self, source, name):
            # py7zr has no per-member lazy-reopen API like zip/tar/rarfile
            # do, so this reads the member fully into memory immediately.
            # Fine for typical file sizes; a very large single member
            # inside a 7z would be read eagerly rather than streamed.
            if not isinstance(source, Path):
                source.seek(0)
            with py7zr.SevenZipFile(source, "r") as zf:
                data = zf.read([name])[name].read()
            return io.BytesIO(data)

    _OPTIONAL_FORMATS.append(SevenZipArchive())
except ImportError:
    pass

try:
    import rarfile

    class RarArchive(ArchiveFormat):
        suffixes = (".rar",)

        def _open(self, source: Union[Path, BinaryIO]) -> "rarfile.RarFile":
            if isinstance(source, Path):
                return rarfile.RarFile(source, "r")
            source.seek(0)
            return rarfile.RarFile(source, "r")

        def list_members(self, source):
            with self._open(source) as rf:
                return [(i.filename, i.file_size) for i in rf.infolist() if not i.is_dir()]

        def open_member(self, source, name):
            rf = self._open(source)
            return _ClosingReader(rf, rf.open(name))

    _OPTIONAL_FORMATS.append(RarArchive())
except ImportError:
    pass


ARCHIVE_FORMATS: dict[str, ArchiveFormat] = {}
for _fmt in (ZipArchive(), TarArchive(), *_OPTIONAL_FORMATS):
    for _suffix in _fmt.suffixes:
        ARCHIVE_FORMATS[_suffix] = _fmt


def _archive_format_for(filename: str) -> Optional[ArchiveFormat]:
    lower = filename.lower()
    for suffix, fmt in ARCHIVE_FORMATS.items():
        if lower.endswith(suffix):
            return fmt
    return None


# ── container formats ────────────────────────────────────────────────────────
#
# Same interface as ArchiveFormat, but registered separately: a zip/tar has
# no extractable content of its own (only its members matter), while an
# .eml *does* -- its subject/body should be indexed under the .eml's own
# virtual_path (via extractors.py's eml extractor) in addition to walking
# its attachments as members. See _handle_source below for how the two
# registries differ in practice.

class EmlAttachments(ArchiveFormat):
    """Treats an .eml's attachments as archive members, so they get
    listed/extracted/recursed exactly like real archive contents (an
    attached .zip, .pdf, or even another .eml all just work). The
    message's own subject/body text is handled separately, by
    extractors.py's eml extractor -- this class only concerns itself with
    what's attached."""
    suffixes = (".eml",)

    def _read_source_bytes(self, source: Union[Path, BinaryIO]) -> bytes:
        if isinstance(source, Path):
            return source.read_bytes()
        source.seek(0)
        return source.read()

    def _attachments(self, source: Union[Path, BinaryIO]) -> list[tuple[str, bytes]]:
        from email import policy
        from email.parser import BytesParser

        msg = BytesParser(policy=policy.default).parsebytes(self._read_source_bytes(source))

        seen: dict[str, int] = {}
        results = []
        unnamed_count = 0

        for part in msg.walk():
            if part.is_multipart():
                continue
            filename = part.get_filename()
            if not filename and part.get_content_disposition() != "attachment":
                continue
            if not filename:
                unnamed_count += 1
                filename = f"attachment-{unnamed_count}"

            # Disambiguate duplicate attachment names (e.g. several
            # "image.png" inline images) so they don't collide as cache
            # keys -- only the virtual_path changes, not the real filename
            # used for extension-based extraction.
            seen[filename] = seen.get(filename, 0) + 1
            if seen[filename] > 1:
                stem, dot, ext = filename.rpartition(".")
                filename = f"{stem}-{seen[filename]}.{ext}" if dot else f"{filename}-{seen[filename]}"

            payload = part.get_payload(decode=True) or b""
            results.append((filename, payload))

        return results

    def list_members(self, source):
        return [(name, len(payload)) for name, payload in self._attachments(source)]

    def open_member(self, source, name):
        for member_name, payload in self._attachments(source):
            if member_name == name:
                return io.BytesIO(payload)
        raise KeyError(f"no such attachment: {name}")


CONTAINER_FORMATS: dict[str, ArchiveFormat] = {}
for _fmt in (EmlAttachments(),):
    for _suffix in _fmt.suffixes:
        CONTAINER_FORMATS[_suffix] = _fmt


def _container_format_for(filename: str) -> Optional[ArchiveFormat]:
    lower = filename.lower()
    for suffix, fmt in CONTAINER_FORMATS.items():
        if lower.endswith(suffix):
            return fmt
    return None


# ── walking ──────────────────────────────────────────────────────────────────

def iter_entries(root: Path, cache_all: bool) -> Iterator[Entry]:
    """Walk `root` on disk, yielding an Entry for every real file, every
    file found inside (possibly nested) archives, and every attachment
    found inside (possibly nested) .eml messages."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if _is_comb_file(path):
            continue

        rel = str(path.relative_to(root))
        if not cache_all and _is_ignored_path(rel):
            continue

        stat = path.stat()
        yield from _handle_source(
            vpath=rel,
            name=path.name,
            mtime=stat.st_mtime,
            size=stat.st_size,
            cache_all=cache_all,
            self_data_func=_make_disk_reader(path),
            # Disk files are cheap to "reopen" -- just reuse the Path, no
            # need to read anything unless recursion actually happens.
            open_for_recursion=lambda path=path: path,
        )


def _make_disk_reader(path: Path) -> Callable[[], Path]:
    def _read() -> Path:
        return path
    return _read


def _make_member_reader(
    fmt: ArchiveFormat, source: Union[Path, BinaryIO], name: str
) -> Callable[[], BinaryIO]:
    def _read() -> BinaryIO:
        return fmt.open_member(source, name)
    return _read


# Members up to this size are buffered fully in memory when descending into
# a nested archive/container; larger ones spill to a spooled temp file.
# Only matters when recursing, since top-level files are read straight off
# disk.
MAX_IN_MEMORY_MEMBER_SIZE = 20 * 1024 * 1024  # 20 MB


def _handle_source(
    vpath: str,
    name: str,
    mtime: float,
    size: int,
    cache_all: bool,
    self_data_func: Callable[[], Union[bytes, Path, BinaryIO]],
    open_for_recursion: Callable[[], Union[Path, BinaryIO]],
) -> Iterator[Entry]:
    """Decide what one file -- on disk, or an archive/eml member -- is, and
    handle it accordingly:

      * a pure archive (zip/tar/...): no Entry of its own, just recurse
        into its members.
      * a hybrid container (.eml): an Entry for the message itself (so
        extractors.py can pull out its subject/body) *and* recursion into
        its attachments.
      * anything else: just a plain leaf Entry.

    `open_for_recursion` is only called when recursion is actually needed,
    so listing a plain file never pays the cost of reading it.
    """
    archive_fmt = _archive_format_for(name)
    container_fmt = None if archive_fmt is not None else _container_format_for(name)

    if archive_fmt is None:
        yield Entry(virtual_path=vpath, mtime=mtime, size=size, data_func=self_data_func)

    fmt = archive_fmt or container_fmt
    if fmt is None:
        return

    try:
        recursion_source = open_for_recursion()
    except Exception:
        return
    yield from _iter_archive(vpath, fmt, recursion_source, mtime, cache_all)


def _iter_archive(
    prefix: str,
    fmt: ArchiveFormat,
    source: Union[Path, BinaryIO],
    mtime: float,
    cache_all: bool,
) -> Iterator[Entry]:
    """Yield an Entry for every member found in `source` via `fmt`,
    recursing into any nested archives/containers found inside it.
    `source` must be reopenable by `fmt` at any time (a Path, or a
    seekable file-like object)."""
    try:
        members = fmt.list_members(source)
    except Exception:
        return

    for name, size in members:
        if not cache_all and _is_ignored_path(name):
            continue

        vpath = f"{prefix}:{name}"
        yield from _handle_source(
            vpath=vpath,
            name=name,
            mtime=mtime,            # inherit the outer archive's mtime
            size=size,
            cache_all=cache_all,
            self_data_func=_make_member_reader(fmt, source, name),
            open_for_recursion=lambda fmt=fmt, source=source, name=name, size=size: (
                _materialize_member(fmt, source, name, size)
            ),
        )


def _materialize_member(
    fmt: ArchiveFormat, source: Union[Path, BinaryIO], name: str, size: int
) -> Union[io.BytesIO, "tempfile.SpooledTemporaryFile"]:
    """Read one archive/container member fully into a buffer so it can
    itself be walked as a nested archive/container. Only called when a
    member's name indicates it needs recursing into (see _handle_source) --
    plain members are never read during the walk, only later and lazily,
    via their Entry.data_func.

    NOTE: the returned buffer is intentionally not closed by the caller.
    Entries yielded from recursing into it hold a lazy reference (their
    data_func reopens it on demand, possibly much later during extraction
    -- see cache.py, which fully drains iter_entries into a list before any
    data_func is called), so closing it right after this function returns
    would leave those entries pointing at a closed buffer. It's released
    by garbage collection once nothing references it anymore.
    """
    with fmt.open_member(source, name) as member_stream:
        if size <= MAX_IN_MEMORY_MEMBER_SIZE:
            return io.BytesIO(member_stream.read())
        buffer = tempfile.SpooledTemporaryFile(max_size=MAX_IN_MEMORY_MEMBER_SIZE)
        shutil.copyfileobj(member_stream, buffer)
        buffer.seek(0)
        return buffer

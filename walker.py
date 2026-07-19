"""
walker.py

Walks a folder on disk and yields one Entry per "logical" file found,
descending into archives recursively.

A file that lives inside an archive gets a virtual_path like:
    archive.zip:inner/folder/file.txt
or, for nested archives:
    outer.zip:inner.zip:deep/file.txt
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
    virtual_path: str          # path used as the cache key / display path
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


# ── walking ──────────────────────────────────────────────────────────────────

def iter_entries(root: Path, cache_all: bool) -> Iterator[Entry]:
    """Walk `root` on disk, yielding an Entry for every real file and every
    file found inside (possibly nested) archives."""
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
        fmt = _archive_format_for(path.name)

        if fmt is not None:
            yield from _iter_archive(rel, fmt, path, stat.st_mtime, cache_all)
        else:
            yield Entry(
                virtual_path=rel,
                mtime=stat.st_mtime,
                size=stat.st_size,
                data_func=_make_disk_reader(path),
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
# a nested archive; larger ones spill to a spooled temp file. Only matters
# for archives-within-archives, since top-level archives are read straight
# off disk.
MAX_IN_MEMORY_MEMBER_SIZE = 20 * 1024 * 1024  # 20 MB


def _iter_archive(
    prefix: str,
    fmt: ArchiveFormat,
    source: Union[Path, BinaryIO],
    mtime: float,
    cache_all: bool,
) -> Iterator[Entry]:
    """Yield an Entry for every file in the archive at `source`, recursing
    into any nested archives found inside it. `source` must be reopenable
    by `fmt` at any time (a Path, or a seekable file-like object)."""
    try:
        members = fmt.list_members(source)
    except Exception:
        return

    for name, size in members:
        if not cache_all and _is_ignored_path(name):
            continue

        vpath = f"{prefix}:{name}"
        inner_fmt = _archive_format_for(name)

        if inner_fmt is not None:
            try:
                with fmt.open_member(source, name) as member_stream:
                    if size <= MAX_IN_MEMORY_MEMBER_SIZE:
                        buffer: Union[io.BytesIO, tempfile.SpooledTemporaryFile] = io.BytesIO(
                            member_stream.read()
                        )
                    else:
                        buffer = tempfile.SpooledTemporaryFile(max_size=MAX_IN_MEMORY_MEMBER_SIZE)
                        shutil.copyfileobj(member_stream, buffer)
                        buffer.seek(0)
            except Exception:
                continue

            # NOTE: `buffer` is intentionally not closed here. Entries
            # yielded from the recursive call below hold a lazy reference
            # to it (their data_func reopens it on demand, possibly much
            # later during extraction -- see cache.py, which fully drains
            # iter_entries into a list before any data_func is called), so
            # closing it as soon as this loop iteration ends would leave
            # those entries pointing at a closed buffer. It's released by
            # garbage collection once nothing references it anymore.
            yield from _iter_archive(vpath, inner_fmt, buffer, mtime, cache_all)
        else:
            yield Entry(
                virtual_path=vpath,
                mtime=mtime,           # inherit the outer archive's mtime
                size=size,
                data_func=_make_member_reader(fmt, source, name),
            )
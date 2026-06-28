"""
walker.py

Walks a folder on disk and yields one Entry per "logical" file found,
descending into .zip archives (and zips-within-zips) transparently.

A file that lives inside a zip gets a virtual_path like:
    archive.zip:inner/folder/file.txt
or, for nested zips:
    outer.zip:inner.zip:deep/file.txt
"""

import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, BinaryIO, Union

CACHE_FILENAME = ".comb_cache.json"

@dataclass
class Entry:
    virtual_path: str          # path used as the cache key / display path
    mtime: float                # modification time, used to detect changes
    size: int                   # size in bytes, used to detect changes
    data_func: Callable[[], Union[bytes, Path, BinaryIO]]


def iter_entries(root: Path) -> Iterator[Entry]:
    """Walk `root` on disk, yielding an Entry for every real file and every
    file found inside (possibly nested) zip archives."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.name == CACHE_FILENAME:
            continue

        rel = str(path.relative_to(root))
        stat = path.stat()

        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    yield from _iter_zip(rel, zf, stat.st_mtime, archive_source=path)
            except (OSError, zipfile.BadZipFile):
                continue
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


class _ZipEntryFile(io.BufferedReader):
    def __init__(self, zf: zipfile.ZipFile, fileobj: BinaryIO):
        super().__init__(fileobj)
        self._zf = zf

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._zf.close()


def _make_zip_reader(source: Union[Path, io.BytesIO, BinaryIO], filename: str) -> Callable[[], BinaryIO]:
    def _read() -> BinaryIO:
        if isinstance(source, Path):
            zf = zipfile.ZipFile(source, "r")
        else:
            try:
                source.seek(0)
            except Exception:
                pass
            zf = zipfile.ZipFile(source, "r")

        return _ZipEntryFile(zf, zf.open(filename))

    return _read


def _iter_zip(prefix: str, zf: zipfile.ZipFile, mtime: float, archive_source: Union[Path, io.BytesIO, BinaryIO, None] = None) -> Iterator[Entry]:
    """Yield an Entry for every file inside the archive, recursing into any
    nested zips found inside it."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        vpath = f"{prefix}:{info.filename}"

        if info.filename.lower().endswith(".zip"):
            try:
                with zf.open(info) as inner_file:
                    yield from _iter_zip_from_file(
                        vpath, inner_file, mtime, info.file_size
                    )
            except Exception:
                continue
        else:
            if archive_source is None:
                raise RuntimeError("archive_source must be provided for zip entry readers")
            yield Entry(
                virtual_path=vpath,
                mtime=mtime,           # inherit outer zip's mtime
                size=info.file_size,
                data_func=_make_zip_reader(archive_source, info.filename),
            )


def _iter_zip_from_file(
    prefix: str,
    fileobj: io.BufferedIOBase,
    mtime: float,
    size: int,
) -> Iterator[Entry]:
    """Walk a nested zip archive from a file-like object without loading it all into RAM."""
    if size <= 20 * 1024 * 1024:
        blob = fileobj.read()
        source = io.BytesIO(blob)
        try:
            with zipfile.ZipFile(source) as inner_zf:
                yield from _iter_zip(prefix, inner_zf, mtime, archive_source=source)
        except zipfile.BadZipFile:
            return
    else:
        tmp = tempfile.SpooledTemporaryFile(max_size=20 * 1024 * 1024)
        shutil.copyfileobj(fileobj, tmp)
        tmp.seek(0)
        try:
            with zipfile.ZipFile(tmp) as inner_zf:
                yield from _iter_zip(prefix, inner_zf, mtime, archive_source=tmp)
        except zipfile.BadZipFile:
            tmp.close()
            return

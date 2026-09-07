"""
Cache.

Schema:
    files       -- one row per indexed file; canonical store of all data
    meta        -- key/value store for cache-wide metadata (built_at, etc.)

"""

import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from walker import iter_entries, INDEX_FILENAME
from extractors import extract_text

FLUSH_EVERY = 50  # rows flushed to DB per batch during build
MAX_TEXT_CHARS = 6_000_000  # truncate extracted text beyond this

_DDL = """
CREATE TABLE IF NOT EXISTS files (
    virtual_path  TEXT PRIMARY KEY,
    mtime         REAL NOT NULL,
    size          INTEGER NOT NULL,
    text          TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    virtual_path,
    text,
    content='files',
    content_rowid='rowid',
    tokenize='trigram case_sensitive 0'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, virtual_path, text)
    VALUES (new.rowid, new.virtual_path, new.text);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, virtual_path, text)
    VALUES ('delete', old.rowid, old.virtual_path, old.text);
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, virtual_path, text)
    VALUES ('delete', old.rowid, old.virtual_path, old.text);
    INSERT INTO files_fts(rowid, virtual_path, text)
    VALUES (new.rowid, new.virtual_path, new.text);
END;
"""


# ── connection ────────────────────────────────────────────────────────────────


def _cache_path(folder: Path) -> Path:
    return Path(folder) / INDEX_FILENAME


@contextmanager
def _connect(folder: Path):
    """Open a WAL-mode connection with the REGEXP function registered."""
    con = sqlite3.connect(_cache_path(folder))
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript(_DDL)
        con.commit()

        # One-time backfill: if files_fts is empty but files has rows, this
        # DB predates the FTS5 triggers -- populate the index from scratch
        # rather than waiting for the next write to each row.
        fts_count = con.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
        files_count = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        if fts_count == 0 and files_count > 0:
            con.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
            con.commit()
        # Register REGEXP so callers can use `text REGEXP ?` in raw SQL if
        # they want; the search() function below uses Python-side filtering
        # instead, but having it in SQL is occasionally handy for debugging.
        con.create_function(
            "REGEXP",
            2,
            lambda pattern, s: bool(re.search(pattern, s or "")),
            deterministic=True,
        )
        yield con
    finally:
        con.close()


# ── public cache API ──────────────────────────────────────────────────────────


def iter_index(folder: Path) -> Iterator[tuple[str, dict]] | None:
    """Yield (virtual_path, meta) pairs one at a time from the cache.

    Returns None if the cache doesn't exist, so callers can distinguish
    "no cache" from "empty cache":

        entries = iter_cache(folder)
        if entries is None:
            rebuild()
        else:
            for path, meta in entries:
                ...
    """

    p = _cache_path(folder)
    if not p.exists():
        return None

    def _generate():
        con = None
        try:
            con = sqlite3.connect(_cache_path(folder))
            con.execute("PRAGMA journal_mode=WAL")
            cur = con.cursor()
            cur.arraysize = 100  # rows fetched from disk per round-trip
            cur.execute("SELECT virtual_path, mtime, size, text FROM files")
            for row in cur:  # cursor is itself an iterator; fetches in arraysize chunks
                yield row[0], {"mtime": row[1], "size": row[2], "text": row[3]}
        finally:
            if con:
                con.close()

    return _generate()


def save_cache(folder: Path, cache: dict) -> None:
    """Bulk-write a plain dict into the DB. Useful for one-off migrations."""
    with _connect(folder) as con:
        con.executemany(
            "INSERT OR REPLACE INTO files (virtual_path, mtime, size, text) VALUES (?, ?, ?, ?)",
            [
                (path, meta["mtime"], meta["size"], meta.get("text", ""))
                for path, meta in cache.get("files", {}).items()
            ],
        )
        con.commit()


def clear_index(folder: Path) -> bool:
    """Delete the cache DB for `folder`."""
    p = _cache_path(folder)
    if p.exists():
        p.unlink()
        return True
    return False


# ── build internals ───────────────────────────────────────────────────────────


def _index_entry(entry, verbose: bool) -> dict:
    if entry.is_container:
        # A pure archive/tar has no text of its own -- only its members
        # do -- so there's nothing to extract, and no need to pay the cost
        # of reading the whole archive via data_func() just to discard it.
        if verbose:
            print(f"  + indexing {entry.virtual_path} ... (container, no text)")
        return {
            "virtual_path": entry.virtual_path,
            "mtime": entry.mtime,
            "size": entry.size,
            "text": "",
        }

    try:
        data = entry.data_func()
        if verbose:
            print(f"  + indexing {entry.virtual_path} ... ")
        text = extract_text(entry.virtual_path, data)
    except Exception:
        text = ""

    if len(text) > MAX_TEXT_CHARS:
        print(
            f"  ! {entry.virtual_path}: text truncated to {MAX_TEXT_CHARS:,} chars "
            f"(was {len(text):,})"
        )
        text = text[:MAX_TEXT_CHARS]

    return {
        "virtual_path": entry.virtual_path,
        "mtime": entry.mtime,
        "size": entry.size,
        "text": text,
    }


def _flush(con: sqlite3.Connection, results: list[dict]) -> None:
    """Write a batch of completed results to `files` and clear the list.
    Does NOT rebuild the FTS index -- that happens once at the end of a
    successful build_cache() so we don't pay the rebuild cost per batch."""
    if not results:
        return
    con.executemany(
        "INSERT OR REPLACE INTO files (virtual_path, mtime, size, text) VALUES (?, ?, ?, ?)",
        [(r["virtual_path"], r["mtime"], r["size"], r["text"]) for r in results],
    )
    con.commit()
    results.clear()


def build_index(
    folder: Path,
    verbose: bool = True,
    index_all: bool = False,
) -> None:
    """(Re)build the cache for `folder`.

    Safe to interrupt: results are flushed every FLUSH_EVERY completions.
    """
    folder = Path(folder)

    with _connect(folder) as con:
        existing = {
            row[0]: (row[1], row[2])
            for row in con.execute(
                "SELECT virtual_path, mtime, size FROM files"
            ).fetchall()
        }

        seen = set()
        count_new = 0
        count_skip = 0
        count_removed = 0

        indexed_results: list[dict] = []
        interrupted = False

        try:
            # Stream entries instead of collecting them all in memory.
            for entry in iter_entries(folder, index_all=index_all, known=existing):
                seen.add(entry.virtual_path)
                prev = existing.get(entry.virtual_path)
                if prev and prev[0] == entry.mtime and prev[1] == entry.size:
                    count_skip += 1
                    continue
                indexed_results.append(_index_entry(entry, verbose))
                if len(indexed_results) >= FLUSH_EVERY:
                    count_new += len(indexed_results)
                    _flush(con, indexed_results)

        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                print(
                    f"\nInterrupted -- saving progress "
                    f"({len(indexed_results)} file(s) buffered, flushing)..."
                )

        finally:
            _flush(con, indexed_results)

            if not interrupted:
                removed = [k for k in existing if k not in seen]
                count_removed = len(removed)
                for k in removed:
                    con.execute("DELETE FROM files WHERE virtual_path = ?", (k,))
                    if verbose:
                        print(f"  - removing {k} from cache")
                if removed:
                    con.commit()

                con.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('built_at', ?)",
                    (str(time.time()),),
                )
                con.commit()

        total = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        if verbose:
            msg = (
                f"Done. {count_new} indexed, {count_skip} unchanged"
                + (f", {count_removed} removed" if count_removed else "")
                + f". Total cached files: {total}."
            )
            print(msg)


def index_exists(folder: Path) -> bool:
    """Return True if a cache exists for `folder`."""
    return _cache_path(folder).exists()


def search_fts(folder: Path, query: str, context: int) -> list[dict]:
    """FTS5 MATCH search. Raises sqlite3.OperationalError on malformed
    FTS5 syntax so callers can decide how to handle it."""
    with _connect(folder) as con:
        cur = con.execute(
            """
            SELECT
                highlight(files_fts, 0, '\x01', '\x02') AS path_hl,
                snippet(files_fts, 1, '\x01', '\x02', '...', ?) AS text_snip,
                bm25(files_fts) AS rank
            FROM files_fts
            WHERE files_fts MATCH ?
            ORDER BY rank
            """,
            (
                2 * context + len(query),
                query,
            ),
        )
        return [
            {"path_hl": row[0], "text_snip": row[1], "rank": row[2]}
            for row in cur.fetchall()
        ]

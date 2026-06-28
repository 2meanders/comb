"""
.comb_cache.json structure:
{
  "built_at": <unix timestamp>,
  "files": {
      "<virtual_path>": {"mtime": ..., "size": ..., "text": "..."},
      ...
  }
}

Rebuilding is incremental: a file is only re-extracted if its mtime/size
have changed since the last build (or --force is passed).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from walker import iter_entries, CACHE_FILENAME
from extractors import extract_text


def _cache_path(folder: Path) -> Path:
    return Path(folder) / CACHE_FILENAME


def load_cache(folder: Path) -> dict | None:
    """Return the cache dict, or None if it doesn't exist / is unreadable."""
    p = _cache_path(folder)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(folder: Path, cache: dict) -> None:
    p = _cache_path(folder)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _index_entry(entry, verbose: bool) -> dict:
    try:
        data = entry.data_func()
        if verbose:
            print(f"  + indexing {entry.virtual_path} ... ")
        text = extract_text(entry.virtual_path, data)
    except Exception:
        text = ""

    return {
        "virtual_path": entry.virtual_path,
        "mtime": entry.mtime,
        "size": entry.size,
        "text": text,
    }


def build_cache(folder: Path, force: bool = False, verbose: bool = True, workers: int = 1) -> None:
    """(Re)build the cache for `folder`. Unless force=True, files whose
    mtime/size haven't changed since the last build are skipped."""
    folder = Path(folder)
    cache = {} if force else (load_cache(folder) or {})
    files = cache.get("files", {})

    seen = set()
    count_new = 0
    count_skip = 0
    count_removed = 0
    pending = []
    indexed_results = []
    try:
        for entry in iter_entries(folder):
            seen.add(entry.virtual_path)
            existing = files.get(entry.virtual_path)

            if (
                existing
                and existing.get("mtime") == entry.mtime
                and existing.get("size") == entry.size
            ):
                count_skip += 1
                continue

            pending.append(entry)

        if workers <= 1:
            for entry in pending:
                if verbose:
                    print(f"  + indexing {entry.virtual_path} ... ")
                indexed_results.append(_index_entry(entry, verbose))
        else:
            worker_count = max(1, min(workers, len(pending) or 1))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                indexed_results = list(executor.map(lambda e: _index_entry(e, verbose=verbose), pending))

    finally:
        count_new += len(indexed_results)
        for item in indexed_results:
            files[item["virtual_path"]] = {
                    "mtime": item["mtime"],
                    "size": item["size"],
                    "text": item["text"],
                }

        removed = [k for k in files if k not in seen]
        count_removed = len(removed)
        for k in removed:
            del files[k]
            if verbose:
                print(f"  - removing {k} from cache")

        # Save the cache
        cache["files"] = files
        cache["built_at"] = time.time()
        save_cache(folder, cache)

    if verbose:
        if count_removed > 0:
            print(
                f"Done. {count_new} indexed, {count_skip} unchanged, "
                f"{count_removed} removed. Total cached files: {len(files)}."
            )
        else:
            print(
                f"Done. {count_new} indexed, {count_skip} unchanged. "
                f"Total cached files: {len(files)}."
            )

def clear_cache(folder: Path) -> bool:
    """Delete the cache file for `folder`. Return True if the cache was deleted, False if it didn't exist."""
    p = _cache_path(folder)
    if p.exists():
        p.unlink()
        return True
    return False

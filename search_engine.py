
import re
from pathlib import Path

from cache import load_cache, build_cache
import sys
import os



def _supports_color() -> bool:
    """Return True if stdout supports ANSI colors.

    Respects the `NO_COLOR` environment convention and ensures stdout is a TTY.
    """
    if "NO_COLOR" in os.environ:
        return False
    return sys.stdout.isatty()


def _color(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def search_cache(folder: Path, term: str, context: int = 40) -> None:
    cache = load_cache(folder)
    if cache is None:
        print("No cache found. Exiting...")
        return
    
    color_enabled = _supports_color()

    try:
        pattern = re.compile(term, re.IGNORECASE)
    except re.error as e:
        print(f"Invalid regex: {e}")
        return
    found_any = False

    for vpath, entry in sorted(cache.get("files", {}).items()):
        vpath_str = str(vpath)
        file_match = pattern.search(vpath_str)

        text = entry.get("text") or ""
        text_match = pattern.search(text) if text else None

        if not file_match and not text_match:
            continue

        found_any = True

        # Build highlighted path (used whether or not there's also a text match)
        parts = []
        last = 0
        for m in pattern.finditer(vpath_str):
            if m.start() > last:
                parts.append(_color(vpath_str[last:m.start()], "1;36", color_enabled))
            parts.append(_color(m.group(0), "1;33", color_enabled))
            last = m.end()
        if last < len(vpath_str):
            parts.append(_color(vpath_str[last:], "1;36", color_enabled))
        path_str = "".join(parts) if file_match else _color(vpath_str, "1;36", color_enabled)

        if text_match:
            # Expand the text snippet regardless of whether the path also matched
            line_no = text.count("\n", 0, text_match.start()) + 1
            start = max(0, text_match.start() - context)
            end = min(len(text), text_match.end() + context)
            snippet = text[start:end]
            snippet = pattern.sub(lambda m: _color(f"{m.group(0)}", "1;33", color_enabled), snippet)
            snippet = " ".join(snippet.split())
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            lineno_str = _color(str(line_no), "1;32", color_enabled)

            if file_match:
                path_hit_str = _color("<path hit>", "1;32", color_enabled)
                print(f"{path_str.strip()}:{path_hit_str} {lineno_str} {prefix} {snippet} {suffix}".strip())
            else:
                print(f"{path_str}:{lineno_str} {prefix} {snippet} {suffix}".strip())
        else:
            # Path matched but no text match (or no text at all)
            path_hit_str = _color("<path hit>", "1;32", color_enabled)
            print(f"{path_str.strip()}:{path_hit_str}")

    if not found_any:
        print("No matches found.")

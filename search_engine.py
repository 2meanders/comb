import re
from pathlib import Path
import sqlite3

from index import iter_index
import sys
import os


def _sgr(code, enabled: bool) -> str:
    """Return a raw SGR escape sequence for `code` (e.g. "1;33"), or the
    reset sequence if code == 0. Returns "" if color is disabled."""
    if not enabled:
        return ""
    return f"\x1b[{code}m"


def _color(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_sgr(code, enabled)}{text}{_sgr(0, enabled)}"


def _supports_color() -> bool:
    """Return True if stdout supports ANSI colors.

    Respects the `NO_COLOR` environment convention and ensures stdout is a TTY.
    """
    if "NO_COLOR" in os.environ:
        return False
    return sys.stdout.isatty()


_REGEX_SIGNALS = re.compile(
    r"""
    \\[dwsbBAZn]     |   # escape sequences: \d \w \s \b etc.
    \[.*\]           |   # character class [...]
    \{\d+(,\d*)?\}   |   # quantifier {3} or {2,5}
    [\^\$]           |   # anchors
    \|               |   # alternation (FTS5 has no | operator)
    \(\?             |   # non-capturing group / lookaround (?:  (?=  (?!
    [^\s\\]\+        |   # literal + used as a quantifier (not standalone)
    [^\s\\]\?            # literal ? used as a quantifier
""",
    re.VERBOSE,
)


def looks_like_regex(term: str) -> bool:
    """Heuristically decide whether `term` is meant as a regex rather than
    an FTS5 MATCH query. Errs toward FTS5 (the fast path) when ambiguous."""
    return bool(_REGEX_SIGNALS.search(term))


def color_hit(vpath: str, snippet: str, hits: list[str]) -> str:
    """Return a string with ANSI color codes highlighting the hits in the snippet."""
    color_enabled = _supports_color()
    # Highlight the hits in the snippet
    for hit in hits:
        snippet = snippet.replace(hit, _color(hit, "1;33", color_enabled))

    # Highlight the virtual path
    suffix_pos = vpath.rfind(":")
    
    vpath_suffix = vpath[:suffix_pos]
    vpath_path = vpath[suffix_pos + 1 :]
    
    vpath_colored = _color(vpath_path, "1;36", color_enabled) + _color(
        vpath_suffix, "1;32", color_enabled
    )

    return f"{vpath_colored}: {snippet}"


def search_index(
    folder: Path, term: str, context: int = 40, mode: str = "auto"
) -> None:

    use_regex = mode == "regex" or (mode == "auto" and looks_like_regex(term))

    if use_regex:
        _search_regex(folder, term, context)
    else:
        _search_fts(folder, term)


def _search_regex(folder: Path, term: str, context: int) -> None:
    index = iter_index(folder)
    if index is None:
        print("No index found. Exiting...")
        return

    color_enabled = _supports_color()
    try:
        pattern = re.compile(term, re.IGNORECASE)
    except re.error as e:
        print(f"Invalid regex: {e}")
        return

    found_any = False
    for vpath, entry in index:
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
                parts.append(_color(vpath_str[last : m.start()], "1;36", color_enabled))
            parts.append(_color(m.group(0), "1;33", color_enabled))
            last = m.end()
        if last < len(vpath_str):
            parts.append(_color(vpath_str[last:], "1;36", color_enabled))
        path_str = (
            "".join(parts) if file_match else _color(vpath_str, "1;36", color_enabled)
        )

        if text_match:
            # Expand the text snippet regardless of whether the path also matched
            line_no = text.count("\n", 0, text_match.start()) + 1
            start = max(0, text_match.start() - context)
            end = min(len(text), text_match.end() + context)
            snippet = text[start:end]
            snippet = pattern.sub(
                lambda m: _color(f"{m.group(0)}", "1;33", color_enabled), snippet
            )
            snippet = " ".join(snippet.split())
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            lineno_str = _color(str(line_no), "1;32", color_enabled)

            if file_match:
                path_hit_str = _color("<path hit>", "1;32", color_enabled)
                print(
                    f"{path_str.strip()}:{path_hit_str} {lineno_str} {prefix} {snippet} {suffix}".strip()
                )
            else:
                print(f"{path_str}:{lineno_str} {prefix} {snippet} {suffix}".strip())
        else:
            # Path matched but no text match (or no text at all)
            path_hit_str = _color("<path hit>", "1;32", color_enabled)
            print(f"{path_str.strip()}:{path_hit_str}")

    if not found_any:
        print("No matches found.")


def _search_fts(folder: Path, term: str) -> None:
    from index import search_fts

    color_enabled = _supports_color()
    try:
        results = search_fts(folder, term)
    except sqlite3.OperationalError as e:
        # Bad FTS5 syntax (e.g. an unmatched quote/paren) -- fall back to regex
        # rather than surfacing a raw sqlite error to the user.
        print(f"FTS query error ({e}); retrying as regex...")
        _search_regex(folder, term, context=40)
        return

    if not results:
        print("No matches found.")
        return

    for r in results:
        path_hl = (
            r["path_hl"]
            .replace("\x01", _sgr("1;33", color_enabled))
            .replace("\x02", _sgr(0, color_enabled))
        )
        text_snip = (
            r["text_snip"]
            .replace("\x01", _sgr("1;33", color_enabled))
            .replace("\x02", _sgr(0, color_enabled))
        )
        file_match = "\x01" in r["path_hl"]
        tag = _color("<path hit>", "1;32", color_enabled) if file_match else ""
        print(f"{path_hl}:{tag} {text_snip}".strip())

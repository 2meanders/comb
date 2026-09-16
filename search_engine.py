import re
import sqlite3
import sys
import os
from pathlib import Path

from index import iter_index

PREVIEW_CHARS = 80  # length of the plain preview shown for path-only hits


def collapse_whitespace(str: str) -> str:
    return " ".join(str.split())


# ── color primitives ──────────────────────────────────────────────────────────


def _supports_color() -> bool:
    """Return True if stdout supports ANSI colors.

    Respects the `NO_COLOR` environment convention and ensures stdout is a TTY.
    """
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _sgr(code, enabled: bool) -> str:
    """Return a raw SGR escape sequence for `code` (e.g. "1;33"), or the
    reset sequence if code == 0. Returns "" if color is disabled."""
    if not enabled:
        return ""
    return f"\x1b[{code}m"


def _color(text: str, code: str, enabled: bool) -> str:
    if not enabled or not text:
        return text
    return f"{_sgr(code, enabled)}{text}{_sgr(0, enabled)}"


# ── shared rendering interface ────────────────────────────────────────────────
#
# Both backends (`_search_regex` and `_search_fts`) ultimately need to print
# one line per hit: a colorized virtual path, and either a colorized snippet
# (when the match is inside the text) or a plain preview of the file's start
# (when only the path matched). These two helpers are the single place that
# knows how that line gets built and colored.


def _colorize_spans(
    text: str,
    spans: list[tuple[int, int]],
    hit_code: str,
    color_enabled: bool,
    base_code: str | None = None,
) -> str:
    """Color the given [start, end) spans in `text` with `hit_code`, and
    optionally color the untouched portions with `base_code`."""
    if not spans:
        return _color(text, base_code, color_enabled) if base_code else text

    parts = []
    last = 0
    for start, end in spans:
        if start > last:
            seg = text[last:start]
            parts.append(_color(seg, base_code, color_enabled) if base_code else seg)
        parts.append(_color(text[start:end], hit_code, color_enabled))
        last = end
    if last < len(text):
        seg = text[last:]
        parts.append(_color(seg, base_code, color_enabled) if base_code else seg)
    return "".join(parts)


def format_hit(
    vpath: str,
    vpath_spans: list[tuple[int, int]],
    snippet: str,
    snippet_spans: list[tuple[int, int]] | None,
    color_enabled: bool,
) -> str:
    """Render one result line. Used by both search backends.

    `snippet_spans=None` means `snippet` is a plain preview (e.g. the start
    of the file for a path-only hit) and should not be highlighted.
    """
    path_str = _colorize_spans(
        vpath, vpath_spans, "1;33", color_enabled, base_code="1;36"
    )
    snippet_str = (
        snippet
        if snippet_spans is None
        else _colorize_spans(snippet, snippet_spans, "1;33", color_enabled)
    )
    snippet_str = collapse_whitespace(snippet_str)

    return f"{path_str}: {snippet_str}".strip()


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Plain (uncolored) preview of the start of `text`."""
    snippet = text[:limit]
    if len(text) > limit:
        snippet += "..."
    return snippet


# ── regex vs. FTS5 routing ────────────────────────────────────────────────────

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


def search_index(
    folder: Path,
    term: str,
    context: int = 40,
    fuzzy_threshold: int = 80,
    mode: str = "auto",
) -> None:
    color_enabled = _supports_color()

    if mode not in ("auto", "fts", "regex", "fuzzy"):
        print(f"Search mode '{mode}' not valid. Falling back to auto.")
        mode = "auto"

    if mode == "auto":
        if looks_like_regex(term):
            _search_regex(folder, term, context, color_enabled)
        else:
            try:
                num_results = _search_fts(folder, term, context, color_enabled)
                if num_results == 0:
                    _search_regex(folder, term, context, color_enabled)
            except sqlite3.OperationalError:
                num_results = _search_regex(folder, term, context, color_enabled)
                if num_results == 0:
                    _search_fuzzy(folder, term, context, fuzzy_threshold, color_enabled)
    elif mode == "regex":
        _search_regex(folder, term, context, color_enabled)
    elif mode == "fts":
        try:
            _search_fts(folder, term, context, color_enabled)
        except sqlite3.OperationalError as e:
            # Bad FTS5 syntax (e.g. an unmatched quote/paren) -- fall back to regex
            # rather than surfacing a raw sqlite error to the user.
            print(f"FTS query error ({e})")
    elif mode == "fuzzy":
        _search_fuzzy(folder, term, context, fuzzy_threshold, color_enabled)


# ── regex backend ─────────────────────────────────────────────────────────────


def _search_regex(folder: Path, term: str, context: int, color_enabled: bool) -> int:
    """
    Returns the number of results.
    """

    index = iter_index(folder)
    if index is None:
        print("No index found. Exiting...")
        return

    try:
        pattern = re.compile(term, re.IGNORECASE)
    except re.error as e:
        print(f"Invalid regex: {e}")
        return

    found_count = 0
    for vpath, entry in index:
        vpath_str = str(vpath)
        vpath_spans = [m.span() for m in pattern.finditer(vpath_str)]

        text = entry.get("text") or ""
        text_match = pattern.search(text) if text else None

        if not vpath_spans and not text_match:
            continue
        found_count += 1

        if text_match:
            start = max(0, text_match.start() - context)
            end = min(len(text), text_match.end() + context)
            raw = text[start:end]
            # Color before collapsing whitespace: escape sequences contain no
            # whitespace, so `" ".join(raw.split())` afterwards is safe and
            # avoids having to re-map match spans onto the collapsed string.
            colored = pattern.sub(
                lambda m: _color(m.group(0), "1;33", color_enabled), raw
            )
            snippet = " ".join(colored.split())
            if start > 0:
                snippet = f"...{snippet}"
            if end < len(text):
                snippet = f"{snippet}..."
            snippet_spans = None  # already colored inline
        else:
            snippet = _preview(text)
            snippet_spans = None

        print(format_hit(vpath_str, vpath_spans, snippet, snippet_spans, color_enabled))

    if found_count > 0:
        print("No matches found.")

    return found_count


# ── FTS5 backend ──────────────────────────────────────────────────────────────


def _search_fts(folder: Path, term: str, context: int, color_enabled: bool) -> int:
    """
    Returns the number of results.
    """
    from index import search_fts

    results = search_fts(folder, term, context)

    if not results:
        print("No matches found.")
        return

    for r in results:
        # highlight()/snippet() already embed \x01..\x02 markers around hits;
        # when there's no match in a column (e.g. a path-only hit has no
        # match inside `text`), snippet() falls back to the start of that
        # column's content -- which gives us the "start of text" preview for
        # free, with no markers to colorize.
        # `highlight()` only wraps matched spans in markers -- it never colors
        # the untouched part of the path. Match the regex backend's look by
        # wrapping the whole path in the base cyan color, and having a hit's
        # closing marker drop back to cyan (not a full reset) so the color
        # continues after the highlighted span instead of going plain.
        base_start = _sgr("1;36", color_enabled)
        hit_start = _sgr("1;33", color_enabled)
        reset = _sgr(0, color_enabled)
        path_str = (
            base_start
            + r["path_hl"].replace("\x01", hit_start).replace("\x02", base_start)
            + reset
        )

        snippet_str = r["text_snip"].replace("\x01", hit_start).replace("\x02", reset)
        snippet_str = collapse_whitespace(snippet_str)
        print(f"{path_str}: {snippet_str}".strip())

    return len(results)


def _search_fuzzy(
    folder: Path, term: str, context: int, threshold: int, color_enabled: bool
) -> int:
    """
    threshold: 0-100
    """
    from rapidfuzz import fuzz

    index = iter_index(folder)
    if index is None:
        print("No index found. Exiting...")
        return 0

    scored = []
    for vpath, entry in index:
        vpath_str = str(vpath)
        text = entry.get("text") or ""

        path_align = fuzz.partial_ratio_alignment(term, vpath_str)
        text_align = fuzz.partial_ratio_alignment(term, text) if text else None
        text_score = text_align.score if text_align else 0

        best_score = max(path_align.score, text_score)
        if best_score < threshold:
            continue

        scored.append((best_score, vpath_str, path_align, text, text_align))

    scored.sort(key=lambda x: x[0], reverse=True)

    for score, vpath_str, path_align, text, text_align in scored:
        vpath_spans = (
            [(path_align.dest_start, path_align.dest_end)]
            if path_align.score >= threshold
            else []
        )

        if text_align and text_align.score >= threshold:
            start = max(0, text_align.dest_start - context)
            end = min(len(text), text_align.dest_end + context)
            snippet = text[start:end]
            snippet_spans = [
                (text_align.dest_start - start, text_align.dest_end - start)
            ]
            if start > 0:
                snippet = "..." + snippet
            snippet_spans = (
                [(s + 3, e + 3) for s, e in snippet_spans]
                if start > 0
                else snippet_spans
            )
        else:
            snippet = _preview(text)
            snippet_spans = None

        print(format_hit(vpath_str, vpath_spans, snippet, snippet_spans, color_enabled))

    if not scored:
        print("No matches found.")
    return len(scored)

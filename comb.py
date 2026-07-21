#!/usr/bin/env python3

"""
comb.py

Archive search tool.

"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

from index import build_index, index_exists, clear_index
from search_engine import search_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_index(folder: Path, verbose: bool):
    cleared = clear_index(folder)
    if verbose:
        print("Index cleared." if cleared else "No index to clear.")


@contextmanager
def interruptible(message: str, folder: Path = None, args=None):
    """
    Wrap a step so Ctrl-C prints a clean message (instead of a traceback)
    and, if --clear was passed, cleans up the index before exiting.
    """
    try:
        yield
    except KeyboardInterrupt:
        print(f"\n{message}")
        if args is not None and getattr(args, "clear", False):
            _clear_index(folder, verbose=args.verbose)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_build(args):
    folder = Path(args.folder)
    with interruptible(
        "Build interrupted by user. Saving partial index.", folder, args
    ):
        build_index(
            folder,
            force=False,
            verbose=args.verbose,
            workers=args.workers,
            index_all=args.all,
        )


def cmd_rebuild(args):
    folder = Path(args.folder)
    with interruptible(
        "Rebuild interrupted by user. Saving partial index.", folder, args
    ):
        build_index(
            folder,
            force=True,
            verbose=args.verbose,
            workers=args.workers,
            index_all=args.all,
        )


def cmd_clear(args):
    folder = Path(args.folder)
    _clear_index(folder, verbose=True)


def cmd_search(args):
    folder = Path(args.folder)

    index_newly_built = False
    with interruptible("Index build interrupted by user. Exiting.", folder, args):
        if not index_exists(folder):
            print("No index found, building one first (this may take a while)...")
            build_index(
                folder, verbose=args.verbose, workers=args.workers, index_all=args.all
            )
            index_newly_built = True

    with interruptible("Index update interrupted by user. Exiting.", folder, args):
        if not args.no_update and not index_newly_built:
            print("Updating index...")
            build_index(
                folder, verbose=args.verbose, workers=args.workers, index_all=args.all
            )

    with interruptible("Search interrupted by user. Exiting.", folder, args):
        print("Searching...")
        search_index(folder, args.term, context=args.context, mode=args.mode)

    if args.clear:
        _clear_index(folder, verbose=args.verbose)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_index_args(p):
    p.add_argument(
        "--folder",
        "-f",
        type=str,
        default=".",
        help="Folder to index (default: current directory)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads to use while indexing (default: 1)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output during indexing",
    )
    p.add_argument(
        "--include-hidden",
        "-a",
        dest="all",
        action="store_true",
        help="Also index hidden/ignored folders and files (e.g. .git). "
        "Normally these are skipped.",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="comb",
        description="Searches an archive with the help of an index.",
        epilog=(
            "Examples:\n"
            "  comb build                     Build an index in the current folder\n"
            "  comb rebuild --folder ~/docs   Rebuild the index from scratch\n"
            "  comb search invoice            Search for 'invoice'\n"
            "  comb search invoice -c 80      ...with more context around matches\n"
            "  comb clear --folder ~/docs     Delete the index\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- build ---
    p_build = subparsers.add_parser(
        "build", help="Build an index (incremental if one already exists)"
    )
    _add_common_index_args(p_build)
    p_build.set_defaults(func=cmd_build)

    # --- rebuild ---
    p_rebuild = subparsers.add_parser(
        "rebuild", help="Rebuild the index from scratch, ignoring existing entries"
    )
    _add_common_index_args(p_rebuild)
    p_rebuild.set_defaults(func=cmd_rebuild)

    # --- clear ---
    p_clear = subparsers.add_parser("clear", help="Delete the index")
    p_clear.add_argument(
        "--folder",
        "-f",
        type=str,
        default=".",
        help="Folder whose index should be cleared (default: current directory)",
    )
    p_clear.set_defaults(func=cmd_clear)

    # --- search ---
    p_search = subparsers.add_parser(
        "search", help="Search the archive (default command)"
    )
    p_search.add_argument("term", help="Term to search for")
    _add_common_index_args(p_search)
    p_search.add_argument(
        "--context",
        "-c",
        type=int,
        default=40,
        help="Characters of context to show around each match (default: 40)",
    )
    p_search.add_argument(
        "-n",
        "--no-update",
        action="store_true",
        help="Skip refreshing the index before searching "
        "(useful if you know the index is already up to date)",
    )
    p_search.add_argument(
        "--clear",
        action="store_true",
        help="Delete the index after this search completes",
    )
    p_search.add_argument(
        "-m",
        "--mode",
        choices=["auto", "fts", "regex"],
        default="auto",
        help="The search mode. Can be either SQLite's FTS5, regex or comb can try to find the best mode.",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main():
    parser = build_parser()

    # Back-compat / convenience: `comb sometext` implies `comb search sometext`
    # if the first arg isn't a known subcommand or option.
    argv = sys.argv[1:]
    known_commands = {
        "build",
        "rebuild",
        "clear",
        "search",
        "-h",
        "--help",
        "--version",
    }
    if argv and argv[0] not in known_commands:
        argv = ["search"] + argv

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

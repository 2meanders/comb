#!/usr/bin/env python3

"""
comb.py

Archive search tool.

Build an index:
    python3 comb.py --build [--folder PATH] [--force]

Search (auto-builds the index the first time):
    python3 comb.py searchterm
    python3 comb.py searchterm --folder /path/to/archive

"""

import argparse
from pathlib import Path

from index import build_index, index_exists, clear_index
from search_engine import search_index


def _clear_index(folder: Path, verbose: bool):
    cleared = clear_index(folder)
    if verbose:
        if cleared:
            print("Index cleared.")
        else:
            print("No index to clear.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Searches the archive with the help of an index."
        )
    )
    parser.add_argument("term", nargs="?", help="Term to search for")
    parser.add_argument(
        "--build", action="store_true",
        help="(Re)build the index instead of searching",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force a full rebuild, ignoring existing index entries",
    )
    parser.add_argument(
        "--context", "-c", type=int, default=40,
        help="Characters of context to show around each match (default: 40)",
    )
    parser.add_argument(
        "-i", "--no-update", action="store_true",
        help="Do not update the index before searching (useful if you know the index is up to date)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show verbose output during indexing",
    )
    parser.add_argument(
        "--folder", "-f", type=str, default=".",
        help="Folder to search (default: current directory)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="If used on its own, it will clear the index if present. If used with a search term, it will clear the index after searching.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker threads to use while indexing (default: 1)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="Show all files. Normally, folders and files like .git are ignored.",
    )


    args = parser.parse_args()
    folder = Path(args.folder)

    if args.build:
        try:
            build_index(folder, force=args.force, verbose=args.verbose, workers=args.workers, index_all=args.all)
        except KeyboardInterrupt:
            print("\nBuild interrupted by user. Saving partial index.")
        return
    
    if args.clear and not args.term:
        _clear_index(folder, verbose=args.verbose)
        return

    if not args.term:
        parser.error("a search term is required unless --build or --clear is given")

    index_newly_built = False
    try:
        if not index_exists(folder):
            print("No index found, building one first (this may take a while)...")
            build_index(folder, verbose=args.verbose, workers=args.workers, index_all=args.all)
            index_newly_built = True
    except KeyboardInterrupt:
        print("\nIndex loading interrupted by user. Exiting.")
        if args.clear:
            _clear_index(folder, verbose=args.verbose)
        return
    
    try:
        if not args.no_update and not index_newly_built:
            print("Updating index...")
            build_index(folder, verbose=args.verbose, workers=args.workers, index_all=args.all)
            print("Searching...")
    except KeyboardInterrupt:
        print("\nIndex update interrupted by user. Exiting.")
        if args.clear:
            _clear_index(folder, verbose=args.verbose)
        return
    
    try:
        search_index(folder, args.term, context=args.context)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting.")
        if args.clear:
            _clear_index(folder, verbose=args.verbose)
        return
    
    if args.clear:
        _clear_index(folder, verbose=args.verbose)


if __name__ == "__main__":
    main()

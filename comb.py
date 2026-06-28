#!/usr/bin/env python3

"""
comb.py

Archive search tool.

Build a cache:
    python3 comb.py --build [--folder PATH] [--force]

Search (auto-builds the cache the first time):
    python3 comb.py searchterm
    python3 comb.py searchterm --folder /path/to/archive

"""

import argparse
from pathlib import Path

from cache import build_cache, load_cache, clear_cache
from search_engine import search_cache


def _clear_cache(folder: Path, verbose: bool):
    cleared = clear_cache(folder)
    if verbose:
        if cleared:
            print("Cache cleared.")
        else:
            print("No cache to clear.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Searches the archive with the help of a cache."
        )
    )
    parser.add_argument("term", nargs="?", help="Term to search for")
    parser.add_argument(
        "--build", action="store_true",
        help="(Re)build the cache instead of searching",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force a full rebuild, ignoring existing cache entries",
    )
    parser.add_argument(
        "--context", "-c", type=int, default=40,
        help="Characters of context to show around each match (default: 40)",
    )
    parser.add_argument(
        "--no-update", action="store_true",
        help="Do not update the cache before searching (useful if you know the cache is up to date)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show verbose output during cache building",
    )
    parser.add_argument(
        "--folder", "-f", type=str, default=".",
        help="Folder to search (default: current directory)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear the cache after building (useful for saving space if you only need to search once). If --clear is used on its own, it will clear the cache without searching.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker threads to use while building the cache (default: 1)",
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="Show all files. Normally, folders and files like .git are ignored.",
    )


    args = parser.parse_args()
    folder = Path(args.folder)

    if args.build:
        try:
            build_cache(folder, force=args.force, verbose=args.verbose, workers=args.workers, cache_all=args.all)
        except KeyboardInterrupt:
            print("\nBuild interrupted by user. Saving partial cache.")
        return
    
    if args.clear and not args.term:
        _clear_cache(folder, verbose=args.verbose)
        return

    if not args.term:
        parser.error("a search term is required unless --build or --clear is given")

    cache_newly_built = False
    try:
        if load_cache(folder) is None:
            print("No cache found, building one first (this may take a while)...")
            build_cache(folder, verbose=args.verbose, workers=args.workers, cache_all=args.all)
            cache_newly_built = True
    except KeyboardInterrupt:
        print("\nCache loading interrupted by user. Exiting.")
        if args.clear:
            _clear_cache(folder, verbose=args.verbose)
        return
    
    try:
        if not args.no_update and not cache_newly_built:
            print("Updating cache...")
            build_cache(folder, verbose=args.verbose, workers=args.workers, cache_all=args.all)
            print("Searching...")
    except KeyboardInterrupt:
        print("\nCache update interrupted by user. Exiting.")
        if args.clear:
            _clear_cache(folder, verbose=args.verbose)
        return
    
    try:
        search_cache(folder, args.term, context=args.context)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting.")
        if args.clear:
            _clear_cache(folder, verbose=args.verbose)
        return
    
    if args.clear:
        _clear_cache(folder, verbose=args.verbose)


if __name__ == "__main__":
    main()

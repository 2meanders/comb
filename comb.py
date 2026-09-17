#!/usr/bin/env python3

"""
comb.py

Archive search tool.

"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import combconfig
from index import build_index, clear_index, index_exists
from search_engine import search_index

# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------
#
# A command's actual settings come from up to two layers, in order of
# precedence (highest first):
#
#   1. Explicit CLI flags (e.g. --folder, --mode, -c)
#   2. A named preset, given with -p/--preset
#
# falling back to comb's built-in defaults for anything neither supplies.
#
# A preset also pins the index's location (its index_dir), which is what
# lets the .combed file live somewhere other than the folder being
# searched.

_BUILTIN_DEFAULTS = {
    "folder": ".",
    "mode": "auto",
    "context": 40,
    "fuzzy_threshold": 80,
    "verbose": False,
    "all": False,  # --include-hidden
    "no_update": False,
    "clear": False,
}

# Preset option keys -> the corresponding attribute name on `args`/the
# merged namespace (mostly the same, except include_hidden <-> all).
_PRESET_TO_ARG = {
    "mode": "mode",
    "context": "context",
    "fuzzy_threshold": "fuzzy_threshold",
    "verbose": "verbose",
    "include_hidden": "all",
    "no_update": "no_update",
    "clear": "clear",
}


def resolve_preset_context(args):
    """Figure out (index_dir, preset_dict) for this invocation.

    If -p/--preset was given, resolve it from the central registry.
    Otherwise returns (None, None) and callers fall back to comb's
    original behavior (index lives inside --folder).
    """
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return None, None
    try:
        index_dir, preset = combconfig.resolve_by_name(preset_name)
    except combconfig.PresetError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if "folder" not in preset:
        print(
            f"Error: preset '{preset_name}' has no folder set. Fix it with:\n"
            f"  comb preset edit {preset_name} --folder <path>"
        )
        sys.exit(1)
    return index_dir, preset


def merge_settings(args, preset: dict | None) -> SimpleNamespace:
    """Combine CLI flags with a preset's defaults and comb's built-in
    defaults, following the precedence described above.

    For flags backed by argparse `store_true` (verbose, all/include-hidden,
    no_update, clear) there's no way to tell "not passed" from "explicitly
    off" -- so a preset can only turn these *on*; passing the flag on the
    CLI can also turn them on, but neither can force one back off. If you
    need it off for one run despite a preset default, override the preset
    with -p omitted or edit the preset.
    """
    preset = preset or {}
    merged = {}
    for key, builtin_default in _BUILTIN_DEFAULTS.items():
        cli_value = getattr(args, key, None)
        preset_key = next((pk for pk, ak in _PRESET_TO_ARG.items() if ak == key), key)

        if isinstance(builtin_default, bool):
            merged[key] = bool(cli_value) or bool(preset.get(preset_key, False))
        elif cli_value is not None:
            merged[key] = cli_value
        elif preset_key in preset:
            merged[key] = preset[preset_key]
        else:
            merged[key] = builtin_default

    # term isn't part of the merge (it's positional, preset-independent)
    if hasattr(args, "term"):
        merged["term"] = args.term

    return SimpleNamespace(**merged)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_index(folder: Path, verbose: bool, index_dir: Path = None):
    cleared = clear_index(folder, index_dir)
    if verbose:
        print("Index cleared." if cleared else "No index to clear.")


@contextmanager
def interruptible(
    message: str, folder: Path = None, index_dir: Path = None, settings=None
):
    """
    Wrap a step so Ctrl-C prints a clean message (instead of a traceback)
    and, if --clear was passed, cleans up the index before exiting.
    """
    try:
        yield
    except KeyboardInterrupt:
        print(f"\n{message}")
        if settings is not None and getattr(settings, "clear", False):
            _clear_index(folder, verbose=settings.verbose, index_dir=index_dir)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_build(args):
    index_dir, preset = resolve_preset_context(args)
    s = merge_settings(args, preset)
    folder = Path(s.folder).expanduser()
    with interruptible(
        "Build interrupted by user. Saving partial index.", folder, index_dir, s
    ):
        build_index(
            folder,
            verbose=s.verbose,
            index_all=s.all,
            index_dir=index_dir,
        )


def cmd_rebuild(args):
    index_dir, preset = resolve_preset_context(args)
    s = merge_settings(args, preset)
    folder = Path(s.folder).expanduser()
    with interruptible(
        "Rebuild interrupted by user. Saving partial index.", folder, index_dir, s
    ):
        _clear_index(folder, s.verbose, index_dir)

        build_index(
            folder,
            verbose=s.verbose,
            index_all=s.all,
            index_dir=index_dir,
        )


def cmd_clear(args):
    index_dir, preset = resolve_preset_context(args)
    s = merge_settings(args, preset)
    folder = Path(s.folder).expanduser()
    _clear_index(folder, verbose=True, index_dir=index_dir)


def cmd_search(args):
    index_dir, preset = resolve_preset_context(args)
    s = merge_settings(args, preset)
    folder = Path(s.folder).expanduser()

    index_newly_built = False
    with interruptible(
        "Index build interrupted by user. Exiting.", folder, index_dir, s
    ):
        if not index_exists(folder, index_dir):
            print("No index found, building one first (this may take a while)...")
            build_index(folder, verbose=s.verbose, index_all=s.all, index_dir=index_dir)
            index_newly_built = True

    with interruptible(
        "Index update interrupted by user. Exiting.", folder, index_dir, s
    ):
        if not s.no_update and not index_newly_built:
            print("Updating index...")
            build_index(folder, verbose=s.verbose, index_all=s.all, index_dir=index_dir)

    with interruptible("Search interrupted by user. Exiting.", folder, index_dir, s):
        print("Searching...")
        search_index(
            folder,
            s.term,
            context=s.context,
            mode=s.mode,
            fuzzy_threshold=s.fuzzy_threshold,
            index_dir=index_dir,
        )

    if s.clear:
        _clear_index(folder, verbose=s.verbose, index_dir=index_dir)


def cmd_preset_add(args):
    options = {
        "mode": args.mode,
        "context": args.context,
        "fuzzy_threshold": args.fuzzy_threshold,
        "verbose": args.verbose or None,
        "include_hidden": args.all or None,
    }
    index_dir = combconfig.add_preset(
        args.name,
        args.folder,
        index_dir=args.index_dir,
        **options,
    )
    print(f"Preset '{args.name}' created.")
    print(f"  folder:    {Path(args.folder).expanduser()}")
    print(f"  index dir: {index_dir}")
    print(f"  Use it with: comb search <term> -p {args.name}")


def cmd_preset_list(args):
    presets = combconfig.list_presets()
    if not presets:
        print("No presets yet. Create one with 'comb preset add'.")
        return
    for name, entry in sorted(presets.items()):
        print(f"{name}")
        print(f"  folder:    {entry.get('folder', '(not set)')}")
        print(f"  index dir: {entry['index_dir']}")


def cmd_preset_show(args):
    try:
        index_dir, preset = combconfig.resolve_by_name(args.name)
    except combconfig.PresetError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"{args.name}")
    for key, value in preset.items():
        print(f"  {key}: {value}")


def cmd_preset_remove(args):
    try:
        index_dir = combconfig.remove_preset(args.name, delete_files=args.delete_files)
    except combconfig.PresetError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Preset '{args.name}' removed.")
    if args.delete_files:
        print(f"  Deleted its index in {index_dir}.")
    else:
        print(f"  Left its index in place at {index_dir}.")


def cmd_preset_edit(args):
    updates = {
        "folder": args.folder,
        "mode": args.mode,
        "context": args.context,
        "fuzzy_threshold": args.fuzzy_threshold,
        "verbose": args.verbose or None,
        "include_hidden": args.all or None,
    }
    try:
        index_dir = combconfig.edit_preset(args.name, **updates)
    except combconfig.PresetError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Preset '{args.name}' updated ({index_dir}).")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_index_args(p):
    p.add_argument(
        "--folder",
        "-f",
        type=str,
        default=None,
        help="Folder to index (default: current directory, or a preset's "
        "folder if one applies)",
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
    p.add_argument(
        "-p",
        "--preset",
        type=str,
        default=None,
        help="Use a named preset (see 'comb preset --help').",
    )


def _add_preset_option_args(p, folder_required=False):
    """Shared flags for `preset add` / `preset edit`: the same knobs a
    preset can pin, so a preset is just 'these flags, saved'."""
    p.add_argument(
        "--folder",
        "-f",
        type=str,
        required=folder_required,
        default=None,
        help="Folder this preset searches (e.g. a Google Drive/OneDrive path)",
    )
    p.add_argument(
        "-m",
        "--mode",
        choices=["auto", "fts", "regex", "fuzzy"],
        default=None,
        help="Default search mode for this preset",
    )
    p.add_argument(
        "--context",
        "-c",
        type=int,
        default=None,
        help="Default context size for this preset",
    )
    p.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=None,
        help="Default fuzzy match threshold for this preset",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Default to verbose output for this preset",
    )
    p.add_argument(
        "--include-hidden",
        "-a",
        dest="all",
        action="store_true",
        help="Default to including hidden/ignored files for this preset",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="comb",
        description="Searches an archive with the help of an index.",
        epilog=(
            "Examples:\n"
            "  comb build                        Build an index in the current folder\n"
            "  comb rebuild --folder ~/docs      Rebuild the index from scratch\n"
            "  comb search invoice               Search for 'invoice'\n"
            "  comb search invoice -c 80         ...with more context around matches\n"
            "  comb clear --folder ~/docs        Delete the index\n"
            "\n"
            "Presets (see 'comb preset --help'):\n"
            "  comb preset add gdrive -f '~/Google Drive'\n"
            "                                     Create a preset with its own index dir\n"
            "  comb search invoice -p gdrive     Search using a named preset\n"
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
        default=None,
        help="Folder whose index should be cleared (default: current directory, "
        "or a preset's folder if one applies)",
    )
    p_clear.add_argument(
        "-p",
        "--preset",
        type=str,
        default=None,
        help="Use a named preset instead of --folder",
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
        default=None,
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
        choices=["auto", "fts", "regex", "fuzzy"],
        default=None,
        help="The search mode. Can be either 'fts' for SQLite's FTS5, 'regex' for regex, 'fuzzy' for fuzzy searching or 'auto' if comb should try to find the best mode.",
    )
    p_search.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=None,
        help="The threshold for a fuzzy search match to be yielded as a result. Does nothing when the search algorithm is not fuzzy search.",
    )
    p_search.set_defaults(func=cmd_search)

    # --- preset ---
    p_preset = subparsers.add_parser(
        "preset",
        help="Manage presets: named shortcuts for a folder + its own index dir",
    )
    preset_sub = p_preset.add_subparsers(dest="preset_command", required=True)

    pp_add = preset_sub.add_parser("add", help="Create a preset")
    pp_add.add_argument("name", help="Name for the preset (used with -p)")
    _add_preset_option_args(pp_add, folder_required=True)
    pp_add.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Where to store this preset's .combed cache "
        "(default: ~/.comb/presets/<name>)",
    )
    pp_add.set_defaults(func=cmd_preset_add)

    pp_list = preset_sub.add_parser("list", help="List presets")
    pp_list.set_defaults(func=cmd_preset_list)

    pp_show = preset_sub.add_parser("show", help="Show a preset's settings")
    pp_show.add_argument("name")
    pp_show.set_defaults(func=cmd_preset_show)

    pp_remove = preset_sub.add_parser("remove", help="Remove a preset")
    pp_remove.add_argument("name")
    pp_remove.add_argument(
        "--delete-files",
        action="store_true",
        help="Also delete its .combed cache from disk "
        "(never touches the searched folder itself)",
    )
    pp_remove.set_defaults(func=cmd_preset_remove)

    pp_edit = preset_sub.add_parser("edit", help="Update a preset's settings")
    pp_edit.add_argument("name")
    _add_preset_option_args(pp_edit, folder_required=False)
    pp_edit.set_defaults(func=cmd_preset_edit)

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
        "preset",
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

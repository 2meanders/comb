"""
combconfig.py

Named presets for comb.

A preset is a short name mapped to:
  * the folder it searches
  * an index_dir -- where its .combed cache file lives, deliberately
    decoupled from the folder being searched, so a cloud-synced folder
    (e.g. Google Drive, OneDrive) can be indexed without a cache file
    being written inside it (and picked up for cloud sync/storage).
  * optional default search/build options (mode, context, etc.)

All presets live in one central registry file, by default
``~/.config/comb/config.toml`` (override with the ``COMB_CONFIG`` env var,
or ``XDG_CONFIG_HOME`` to move the default config directory). Presets are
looked up only by name, via ``-p/--preset`` -- there's no directory-based
auto-detection.
"""

import os
import tomllib
from pathlib import Path

# Options a preset is allowed to pin defaults for. Kept in sync with the
# CLI flags in comb.py that these can override.
OPTION_KEYS = (
    "mode",
    "context",
    "fuzzy_threshold",
    "verbose",
    "include_hidden",
    "no_update",
    "clear",
)


class PresetError(Exception):
    """Raised for any user-facing preset problem (unknown name, malformed
    registry file, etc.) so comb.py can print a clean message instead of a
    traceback."""


# ---------------------------------------------------------------------------
# Minimal TOML writer
# ---------------------------------------------------------------------------
# comb's own registry file is simple enough (one table per preset, scalar
# values) that a small hand-rolled writer avoids adding a dependency just
# to pair with the stdlib's read-only `tomllib`.


def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_registry_toml(path: Path, presets: dict[str, dict]) -> None:
    lines = []
    for name, kv in sorted(presets.items()):
        lines.append(f"[presets.{name}]")
        for key, value in kv.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Registry file
# ---------------------------------------------------------------------------


def registry_path() -> Path:
    if "COMB_CONFIG" in os.environ:
        return Path(os.environ["COMB_CONFIG"]).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "comb" / "config.toml"


def load_registry() -> dict[str, dict]:
    """Return {name: {folder, index_dir, ...}} for every preset. Empty
    dict if no registry file exists yet."""
    path = registry_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise PresetError(f"Could not parse {path}: {e}")

    raw_presets = data.get("presets", {})
    presets, migrated_any = _migrate_legacy_entries(raw_presets)
    if migrated_any:
        try:
            save_registry(presets)
        except OSError:
            pass  # best effort -- don't fail a read just because we can't write
    return presets


def _migrate_legacy_entries(raw_presets: dict) -> tuple[dict, bool]:
    """An earlier version of comb stored the registry as `name -> index_dir`
    (a bare string), with each preset's folder/options in a separate
    `.combconfig` file inside that directory. Fold any such entries into
    today's `name -> {folder, index_dir, ...}` shape, recovering the
    folder/options from that old .combconfig when it's still there.
    """
    migrated = {}
    changed = False
    for name, raw in raw_presets.items():
        if isinstance(raw, dict):
            migrated[name] = raw
            continue

        changed = True
        index_dir = Path(raw).expanduser()
        entry = {"index_dir": str(index_dir)}

        legacy_config = index_dir / ".combconfig"
        if legacy_config.is_file():
            try:
                with open(legacy_config, "rb") as f:
                    legacy_data = tomllib.load(f)
                legacy_preset = legacy_data.get("preset", {})
                entry.update({k: v for k, v in legacy_preset.items() if k != "name"})
            except (tomllib.TOMLDecodeError, OSError):
                pass

        if "folder" not in entry:
            print(
                f"Note: preset '{name}' was in an old format and its folder "
                f"could not be recovered automatically. Set it with:\n"
                f"  comb preset edit {name} --folder <path>"
            )

        migrated[name] = entry

    return migrated, changed


def save_registry(presets: dict[str, dict]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_registry_toml(path, presets)


# ---------------------------------------------------------------------------
# High-level preset operations, used by the `comb preset ...` subcommands
# ---------------------------------------------------------------------------


def default_index_home(name: str) -> Path:
    """Where a preset's index lives if the user doesn't specify --index-dir."""
    return Path.home() / ".comb" / "presets" / name


def add_preset(
    name: str, folder: str, index_dir: str | Path | None = None, **options
) -> Path:
    """Create (or overwrite) a preset in the registry. Returns the
    resolved index_dir."""
    resolved_dir = (
        Path(index_dir).expanduser() if index_dir else default_index_home(name)
    )
    resolved_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "folder": str(Path(folder).expanduser()),
        "index_dir": str(resolved_dir),
    }
    for key, value in options.items():
        if key in OPTION_KEYS and value is not None:
            entry[key] = value

    registry = load_registry()
    registry[name] = entry
    save_registry(registry)
    return resolved_dir


def edit_preset(name: str, **updates) -> Path:
    """Update fields of an existing preset."""
    registry = load_registry()
    if name not in registry:
        raise PresetError(
            f"No preset named '{name}'. Run 'comb preset list' to see what's available."
        )
    entry = registry[name]

    for key, value in updates.items():
        if value is None:
            continue
        if key in ("folder", "index_dir"):
            entry[key] = str(Path(value).expanduser())
        elif key in OPTION_KEYS:
            entry[key] = value

    registry[name] = entry
    save_registry(registry)
    return Path(entry["index_dir"]).expanduser()


def remove_preset(name: str, delete_files: bool = False) -> Path:
    """Remove `name` from the registry. If delete_files is True, also
    removes its .combed cache from disk (never touches the searched
    folder itself)."""
    registry = load_registry()
    entry = registry.pop(name, None)
    if entry is None:
        raise PresetError(f"No preset named '{name}'.")
    save_registry(registry)

    index_dir = Path(entry["index_dir"]).expanduser()
    if delete_files:
        from walker import INDEX_FILENAME

        cache_file = index_dir / INDEX_FILENAME
        if cache_file.exists():
            cache_file.unlink()
        try:
            index_dir.rmdir()  # only succeeds if now empty
        except OSError:
            pass
    return index_dir


def list_presets() -> dict[str, dict]:
    """Return {name: {folder, index_dir (as Path), ...}} for every preset."""
    result = {}
    for name, entry in load_registry().items():
        result[name] = {**entry, "index_dir": Path(entry["index_dir"]).expanduser()}
    return result


def resolve_by_name(name: str) -> tuple[Path, dict]:
    """Look up a preset by name. Returns (index_dir, preset_dict)."""
    registry = load_registry()
    if name not in registry:
        raise PresetError(
            f"No preset named '{name}'. Run 'comb preset list' to see what's available."
        )
    entry = registry[name]
    return Path(entry["index_dir"]).expanduser(), entry

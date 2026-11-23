r"""See what was changed in an older copy of this project, before replacing it with this one.

    python scripts/compare_with_release.py "D:\path\to\the older copy"
    python scripts/compare_with_release.py "D:\path\to\the older copy" --original docqa-code-2026-11-20.zip

A copy that was handed over as a zip and then edited (by hand, or by an assistant such
as Copilot) holds changes that this version does not have. This script lists them, so
that each one is carried over on purpose instead of being lost:

- files that were changed, added or deleted since the release the copy started from;
- values in its config.yaml that differ from that release's.

With --original (the zip, or the folder, of that release as it was handed over) it also
prints what exactly was changed in each file.

It only reads. Neither folder is changed. Folders that hold generated or downloaded
content (the virtual environment, data, models, docs) are not looked at.

How it knows what a release contained: scripts/releases/<date>.json lists every file of
that release with a hash of its content (line endings ignored), and holds the release's
config.yaml as data. "--write-manifest <date>" writes such a file from a git commit; it
is run once, when a release is packaged.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASES_DIR = Path(__file__).resolve().parent / "releases"
CONFIG_FILE = "config.yaml"

# Generated or downloaded content: never part of a release, so never compared.
SKIPPED_AT_THE_TOP = frozenset(
    {".git", ".venv", "venv", "env", "data", "models", "docs", "dist", ".ruff_cache", ".mypy_cache", ".idea"}
)
SKIPPED_ANYWHERE = frozenset({"__pycache__", ".pytest_cache", "node_modules"})
SKIPPED_PREFIXES = ("eval/results/", "scripts/releases/")
SKIPPED_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True)
class Comparison:
    """How a folder differs from a release. Paths use forward slashes."""

    release: str
    changed: tuple[str, ...]
    added: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def differences(self) -> int:
        return len(self.changed) + len(self.missing)


# -- hashing and walking ----------------------------------------------------------------


def content_hash(data: bytes) -> str:
    """SHA-256 of a file's content. For text, line endings do not count.

    A zip made on Windows and a file saved by an editor differ in line endings alone
    (CRLF or LF). That is not a change worth reporting.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return hashlib.sha256(data).hexdigest()
    normal = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()


def is_compared(path: str) -> bool:
    """False for a path that holds generated or downloaded content."""
    parts = PurePosixPath(path).parts
    if not parts or parts[0] in SKIPPED_AT_THE_TOP:
        return False
    if any(part in SKIPPED_ANYWHERE or part.endswith(".egg-info") for part in parts):
        return False
    return not path.startswith(SKIPPED_PREFIXES) and not path.endswith(SKIPPED_SUFFIXES)


def hash_folder(folder: Path) -> dict[str, str]:
    """The content hash of every compared file under ``folder``."""
    found: dict[str, str] = {}
    for path in _walk(folder, folder):
        found[path.relative_to(folder).as_posix()] = content_hash(path.read_bytes())
    return found


def _walk(folder: Path, root: Path) -> Iterator[Path]:
    for entry in sorted(folder.iterdir()):
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir():
            # A folder is skipped as a whole, so a virtual environment is never walked.
            if is_compared(relative + "/x"):
                yield from _walk(entry, root)
        elif entry.is_file() and is_compared(relative):
            yield entry


# -- manifests --------------------------------------------------------------------------


def load_manifests(folder: Path = RELEASES_DIR) -> dict[str, dict[str, Any]]:
    """Every release manifest in ``folder``, by release name."""
    manifests = {}
    for path in sorted(folder.glob("*.json")):
        manifests[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def write_manifest(release: str, commit: str, folder: Path = RELEASES_DIR, root: Path = PROJECT_ROOT) -> Path:
    """Write the manifest of the files in a git commit. Needs git; used when packaging."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout

    names = [name for name in git("ls-tree", "-r", "--name-only", "-z", commit).decode("utf-8").split("\0") if name]
    files = {name: content_hash(git("show", f"{commit}:{name}")) for name in names if is_compared(name)}
    config = yaml.safe_load(git("show", f"{commit}:{CONFIG_FILE}").decode("utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{release}.json"
    manifest = {
        "release": release,
        "commit": git("rev-parse", "--short", commit).decode("ascii").strip(),
        "files": dict(sorted(files.items())),
        "config": config,
    }
    target.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return target


# -- comparing --------------------------------------------------------------------------


def compare(files: Mapping[str, str], release: str, manifest: Mapping[str, Any]) -> Comparison:
    """Which of ``files`` differ from the release, which are new, and which are gone."""
    shipped: Mapping[str, str] = manifest["files"]
    return Comparison(
        release=release,
        changed=tuple(sorted(path for path in files if path in shipped and files[path] != shipped[path])),
        added=tuple(sorted(path for path in files if path not in shipped)),
        missing=tuple(sorted(path for path in shipped if path not in files)),
    )


def closest_release(files: Mapping[str, str], manifests: Mapping[str, Mapping[str, Any]]) -> Comparison:
    """The comparison with the release the folder most likely started from.

    That is the release from which it differs in the fewest files. With equally many,
    the earlier release wins.
    """
    if not manifests:
        raise ValueError("There is no release manifest to compare with.")
    comparisons = [compare(files, name, manifests[name]) for name in sorted(manifests)]
    return min(comparisons, key=lambda comparison: comparison.differences)


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Nested settings as ``{"section.key": value}``. Lists count as one value."""
    if not isinstance(data, dict):
        return {prefix: data} if prefix else {}
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


def config_differences(shipped: Any, theirs: Any) -> list[tuple[str, str, str]]:
    """``(setting, value in the release, value in the copy)`` for every setting that differs.

    A setting that one side does not have is shown as ``(not there)``.
    """
    before, after = flatten(shipped), flatten(theirs)
    absent = "(not there)"
    return [
        (key, _show(before.get(key, absent)), _show(after.get(key, absent)))
        for key in sorted(before.keys() | after.keys())
        if before.get(key, absent) != after.get(key, absent)
    ]


def _show(value: Any) -> str:
    return value if value == "(not there)" else json.dumps(value, ensure_ascii=False, default=str)


# -- the release as it was handed over ----------------------------------------------------


class Original:
    """The files of a release as handed over: a folder, or the zip itself."""

    def __init__(self, source: Path) -> None:
        self._folder: Path | None = None
        self._zip: zipfile.ZipFile | None = None
        self._names: dict[str, str] = {}
        if source.is_dir():
            self._folder = _project_folder(source)
        else:
            self._zip = zipfile.ZipFile(source)
            names = [name for name in self._zip.namelist() if not name.endswith("/")]
            # A zip made with a prefix holds everything under one folder, e.g. "docqa/".
            tops = {name.split("/", 1)[0] for name in names}
            strip = len(next(iter(tops))) + 1 if len(tops) == 1 and all("/" in name for name in names) else 0
            self._names = {name[strip:]: name for name in names}

    def read(self, path: str) -> bytes | None:
        if self._folder is not None:
            file = self._folder / path
            return file.read_bytes() if file.is_file() else None
        assert self._zip is not None
        name = self._names.get(path)
        return self._zip.read(name) if name else None

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def text_diff(path: str, before: bytes, after: bytes, max_lines: int) -> list[str]:
    """What changed in one file, as a unified diff. Says so if the file is not text."""
    try:
        old = before.decode("utf-8").replace("\r\n", "\n").splitlines()
        new = after.decode("utf-8").replace("\r\n", "\n").splitlines()
    except UnicodeDecodeError:
        return [f"    ({path} is not a text file, so its changes cannot be shown.)"]
    lines = list(difflib.unified_diff(old, new, f"release/{path}", f"copy/{path}", lineterm="", n=2))
    shown = [f"    {line}" for line in lines[:max_lines]]
    if len(lines) > max_lines:
        shown.append(f"    ... {len(lines) - max_lines} more line(s). Use --max-lines to see them.")
    return shown


# -- the report -------------------------------------------------------------------------


def _project_folder(folder: Path) -> Path:
    """``folder``, or the one folder inside it that holds the project."""
    if (folder / CONFIG_FILE).is_file():
        return folder
    inside = [entry for entry in folder.iterdir() if entry.is_dir() and (entry / CONFIG_FILE).is_file()]
    return inside[0] if len(inside) == 1 else folder


def report(
    folder: Path,
    original: Original | None = None,
    max_lines: int = 80,
    manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """The comparison of one older copy with the release it started from, as lines of text.

    ``manifests`` defaults to the ones in scripts/releases.
    """
    folder = _project_folder(folder)
    manifests = manifests if manifests is not None else load_manifests()
    result = closest_release(hash_folder(folder), manifests)
    manifest = manifests[result.release]
    lines = [
        f"Compared: {folder}",
        f"With:     release {result.release} ({len(manifest['files'])} files), the one it differs from least.",
        "",
    ]

    def listing(title: str, paths: Sequence[str]) -> None:
        lines.append(f"{title}: {len(paths) or 'none'}")
        lines.extend(f"    {path}" for path in paths)

    listing("Changed since that release", result.changed)
    listing("Added", result.added)
    listing("Deleted", result.missing)

    lines.append("")
    their_config = folder / CONFIG_FILE
    if not their_config.is_file():
        lines.append(f"{CONFIG_FILE}: not found in the copy.")
    else:
        try:
            theirs = yaml.safe_load(their_config.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            lines.append(f"{CONFIG_FILE}: could not be read as YAML ({type(exc).__name__}).")
        else:
            differences = config_differences(manifest["config"], theirs)
            lines.append(f"Settings in {CONFIG_FILE} that differ from the release: {len(differences) or 'none'}")
            lines.extend(f"    {key}: {before}  ->  {after}" for key, before, after in differences)

    if original is not None and result.changed:
        lines += ["", "What was changed in each file:"]
        for path in result.changed:
            before = original.read(path)
            lines.append(f"  {path}")
            if before is None:
                lines.append("    (not found in the original, so its changes cannot be shown.)")
            else:
                lines.extend(text_diff(path, before, (folder / path).read_bytes(), max_lines))
    elif result.changed:
        lines += ["", "Add --original <the release's zip or folder> to see what was changed in each file."]

    lines += [
        "",
        "Before the copy is replaced by this version:",
        "  - each changed setting: set it again in this version's config.yaml, if it is still wanted;",
        "  - each changed or added file: carry the change over by hand, into this version's file;",
        "  - nothing here is copied or overwritten by this script.",
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/compare_with_release.py",
        description="List what was changed in an older copy of this project since it was handed over.",
    )
    parser.add_argument("folder", nargs="?", help="the older copy of the project")
    parser.add_argument("--original", help="the zip or folder of the release as handed over, to show the changes")
    parser.add_argument("--max-lines", type=int, default=80, help="longest change shown per file (default 80)")
    parser.add_argument("--write-manifest", metavar="DATE", help="write scripts/releases/DATE.json (needs git)")
    parser.add_argument("--commit", default="HEAD", help="with --write-manifest: the git commit (default HEAD)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.write_manifest:
        target = write_manifest(args.write_manifest, args.commit)
        print(f"Wrote {target}")
        return 0
    if not args.folder:
        parser.error("name the folder of the older copy")
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 1
    if folder.resolve() == PROJECT_ROOT:
        print("That is this version's own folder. Name the folder of the older copy.")
        return 1

    original = None
    if args.original:
        source = Path(args.original)
        if not source.exists():
            print(f"Not found: {source}")
            return 1
        try:
            original = Original(source)
        except zipfile.BadZipFile:
            print(f"Neither a folder nor a zip file: {source}")
            return 1
    try:
        print("\n".join(report(folder, original, args.max_lines)))
    except ValueError as exc:
        print(exc)
        return 1
    finally:
        if original is not None:
            original.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

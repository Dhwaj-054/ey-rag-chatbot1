"""scripts/compare_with_release.py: what was changed in an older copy of the project."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "compare_with_release.py"
_spec = importlib.util.spec_from_file_location("compare_with_release", SCRIPT)
tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tool  # dataclasses look their module up by name
_spec.loader.exec_module(tool)

RELEASE_FILES = {
    "config.yaml": "retrieval:\n  candidates: 30\n  rerank_score_threshold: null\npaths:\n  docs_dir: docs\n",
    "README.md": "# Project\n\nHow to run it.\n",
    "src/docqa/llm.py": "TIMEOUT = 600\n\n\ndef ask():\n    return 'answer'\n",
    "tests/test_llm.py": "def test_ask():\n    assert True\n",
}


def write_files(folder: Path, files: dict[str, str], newline: str = "\n") -> Path:
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return folder


def manifest_of(files: dict[str, str]) -> dict:
    import yaml

    return {
        "release": "2026-01-01",
        "commit": "abc1234",
        "files": {name: tool.content_hash(text.encode("utf-8")) for name, text in files.items()},
        "config": yaml.safe_load(files["config.yaml"]),
    }


@pytest.fixture
def manifests() -> dict:
    return {"2026-01-01": manifest_of(RELEASE_FILES)}


# -- hashing and what is looked at --------------------------------------------------------


def test_line_endings_alone_are_not_a_change():
    assert tool.content_hash(b"one\r\ntwo\r\n") == tool.content_hash(b"one\ntwo\n")
    assert tool.content_hash(b"one\ntwo\n") != tool.content_hash(b"one\ntwo!\n")


def test_a_file_that_is_not_text_is_hashed_as_it_is():
    picture = b"\x89PNG\r\n\x1a\n\xff\xfe"

    assert tool.content_hash(picture) != tool.content_hash(picture.replace(b"\r\n", b"\n"))


@pytest.mark.parametrize(
    ("path", "compared"),
    [
        ("config.yaml", True),
        ("src/docqa/llm.py", True),
        ("src/docqa/web/static/app.js", True),
        ("notes/models.md", True),
        (".venv/Lib/site-packages/torch/__init__.py", False),
        ("data/qdrant/meta.json", False),
        ("models/docling/layout/model.safetensors", False),
        ("docs/Master Directions/KYC.pdf", False),
        ("src/docqa/__pycache__/llm.cpython-313.pyc", False),
        ("src/docqa.egg-info/PKG-INFO", False),
        ("eval/results/2026-11-21/summary.csv", False),
        ("eval/questions.csv", True),
        ("scripts/releases/2026-11-20.json", False),
        ("src/docqa/models/thing.py", True),  # only the top-level models folder is generated
    ],
)
def test_generated_and_downloaded_content_is_not_compared(path, compared):
    assert tool.is_compared(path) is compared


def test_a_folder_is_hashed_without_walking_into_the_skipped_ones(tmp_path):
    write_files(tmp_path, RELEASE_FILES)
    write_files(tmp_path, {".venv/Lib/huge.py": "x\n", "data/parsed/a.json": "{}\n", "src/docqa/__pycache__/a.pyc": "x"})

    assert sorted(tool.hash_folder(tmp_path)) == sorted(RELEASE_FILES)


# -- comparing ------------------------------------------------------------------------------


def test_changed_added_and_deleted_files_are_told_apart(tmp_path, manifests):
    files = dict(RELEASE_FILES)
    files["src/docqa/llm.py"] = files["src/docqa/llm.py"].replace("600", "900")
    files["notes/from_the_other_laptop.md"] = "It ran.\n"
    del files["tests/test_llm.py"]
    write_files(tmp_path, files, newline="\r\n")  # saved on Windows: still only one real change

    result = tool.compare(tool.hash_folder(tmp_path), "2026-01-01", manifests["2026-01-01"])

    assert result.changed == ("src/docqa/llm.py",)
    assert result.added == ("notes/from_the_other_laptop.md",)
    assert result.missing == ("tests/test_llm.py",)


def test_a_copy_is_compared_with_the_release_it_differs_from_least(tmp_path, manifests):
    later = dict(RELEASE_FILES, **{"src/docqa/llm.py": "TIMEOUT = 30\n", "src/docqa/web.py": "PORT = 8000\n"})
    manifests["2026-02-02"] = manifest_of(later)
    write_files(tmp_path, dict(later, **{"README.md": "# Project\n\nEdited.\n"}))

    result = tool.closest_release(tool.hash_folder(tmp_path), manifests)

    assert result.release == "2026-02-02"
    assert result.changed == ("README.md",)
    with pytest.raises(ValueError, match="no release manifest"):
        tool.closest_release({}, {})


def test_settings_that_differ_are_named_with_both_values():
    shipped = {"retrieval": {"candidates": 30, "rerank_score_threshold": None}, "paths": {"docs_dir": "docs"}}
    theirs = {
        "retrieval": {"candidates": 30, "rerank_score_threshold": 0.5},
        "paths": {"docs_dir": "D:/docs"},
        "extra": {"added_by_hand": True},
    }

    assert tool.config_differences(shipped, theirs) == [
        ("extra.added_by_hand", "(not there)", "true"),
        ("paths.docs_dir", '"docs"', '"D:/docs"'),
        ("retrieval.rerank_score_threshold", "null", "0.5"),
    ]
    assert tool.config_differences(shipped, shipped) == []


# -- the report -----------------------------------------------------------------------------


def edited_copy(folder: Path) -> Path:
    files = dict(RELEASE_FILES)
    files["config.yaml"] = files["config.yaml"].replace("rerank_score_threshold: null", "rerank_score_threshold: 0.5")
    files["src/docqa/llm.py"] = files["src/docqa/llm.py"].replace("600", "900")
    return write_files(folder, files)


def test_the_report_lists_files_and_settings_and_changes_nothing(tmp_path, manifests):
    copy = edited_copy(tmp_path / "copy" / "docqa")
    before = tool.hash_folder(copy)

    lines = tool.report(tmp_path / "copy", manifests=manifests)  # the folder that holds the project folder
    text = "\n".join(lines)

    assert "Changed since that release: 2" in text
    assert "    src/docqa/llm.py" in lines and "    config.yaml" in lines
    assert "Added: none" in lines and "Deleted: none" in lines
    assert "    retrieval.rerank_score_threshold: null  ->  0.5" in lines
    assert "Add --original" in text
    assert tool.hash_folder(copy) == before


def test_with_the_original_the_report_shows_what_was_changed(tmp_path, manifests):
    copy = edited_copy(tmp_path / "copy")
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, text in RELEASE_FILES.items():
            zipped.writestr(f"docqa/{name}", text.replace("\n", "\r\n"))  # under one folder, as git archive does

    for source in (archive, write_files(tmp_path / "unpacked", RELEASE_FILES)):
        original = tool.Original(source)
        try:
            text = "\n".join(tool.report(copy, original, manifests=manifests))
        finally:
            original.close()

        assert "    -TIMEOUT = 600" in text and "    +TIMEOUT = 900" in text
        assert "    +  rerank_score_threshold: 0.5" in text
        assert "    --- release/src/docqa/llm.py" in text and "    +++ copy/src/docqa/llm.py" in text


def test_a_long_change_is_cut_and_says_so():
    before = "\n".join(f"line {n}" for n in range(60)).encode()
    after = "\n".join(f"changed {n}" for n in range(60)).encode()

    shown = tool.text_diff("a.py", before, after, max_lines=10)

    assert len(shown) == 11 and "more line(s)" in shown[-1]
    assert tool.text_diff("a.png", b"\xff\xfe", b"\xff\xfd", 10) == [
        "    (a.png is not a text file, so its changes cannot be shown.)"
    ]


# -- the command line and the manifests that ship ---------------------------------------------


def test_the_first_release_has_a_manifest():
    manifests = tool.load_manifests()

    first = manifests["2026-11-20"]
    assert first["commit"] == "62876b3"
    assert {"config.yaml", "src/docqa/llm.py", "README.md", "HANDOVER.md"} <= set(first["files"])
    assert first["config"]["retrieval"]["candidates"] == 30
    assert first["config"]["retrieval"]["rerank_score_threshold"] is None
    assert not any(name.startswith(("data/", "docs/", "dist/")) for name in first["files"])


def test_command_line_reports_and_refuses_what_it_cannot_compare(tmp_path, capsys):
    copy = edited_copy(tmp_path / "copy")

    assert tool.main([str(copy)]) == 0
    assert "Changed since that release" in capsys.readouterr().out

    assert tool.main([str(tmp_path / "nowhere")]) == 1
    assert "Not a folder" in capsys.readouterr().out
    assert tool.main([str(PROJECT_ROOT)]) == 1
    assert "this version's own folder" in capsys.readouterr().out
    assert tool.main([str(copy), "--original", str(tmp_path / "missing.zip")]) == 1
    assert "Not found" in capsys.readouterr().out

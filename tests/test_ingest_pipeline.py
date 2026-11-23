"""Ingestion from start to finish.

Docling's conversion of a PDF is replaced by hand-built Docling documents
(tests/docling_samples.py), chosen by file name. Everything after that is real: the
parse cache, element extraction, the personal-data check, chunking, the SQLite
register and the Qdrant store. The embedding model and the vision model are fakes.
"""

from __future__ import annotations

import json

import pytest

from docqa.db import Database
from docqa.ingest import ingest_path
from docqa.ingest.__main__ import main
from docqa.ingest.documents import doc_id_for
from docqa.ingest.parser import ParseError, cache_paths
from docqa.ingest.pipeline import IngestAborted, find_documents
from docqa.retrieve import retrieve
from docqa.schema import ElementType
from docqa.tokens import WordCounter
from tests.docling_samples import lending_policy, with_identifiers, with_table_of_contents
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedLLM

FIGURE_REPLY = json.dumps(
    {
        "figure_type": "flowchart",
        "title": "Loan application process",
        "nodes": ["Application received", "Credit score check", "Approve"],
        "edges": [{"source": "Application received", "target": "Credit score check", "condition": None}],
    }
)


@pytest.fixture
def conversions(monkeypatch):
    """Replace Docling's PDF conversion. Returns the list of files it was asked to convert."""
    calls: list[str] = []

    def convert(path, config):
        calls.append(path.name)
        if "broken" in path.name:
            raise ParseError(f"Docling could not parse {path.name}: the file is damaged")
        if "customers" in path.name:
            return with_identifiers(5)
        if "contents" in path.name:
            return with_table_of_contents()
        return lending_policy()

    monkeypatch.setattr("docqa.ingest.parser.convert_pdf", convert)
    return calls


class Bench:
    """One temporary project: a docs folder, a store, a database and the fakes."""

    def __init__(self, config, store) -> None:
        self.config = config
        self.store = store
        self.db = Database(config.paths.sqlite_path)
        self.llm = ScriptedLLM(config, FIGURE_REPLY)
        self.lines: list[str] = []
        config.paths.docs_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, content: bytes = b"%PDF-1.7 first version"):
        path = self.config.paths.docs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def run(self, path=None, config=None, **options):
        config = config or self.config
        return ingest_path(
            path or config.paths.docs_dir,
            config,
            self.llm.client,
            embedder=FakeEmbedder(),
            store=self.store,
            db=self.db,
            counter=WordCounter(),
            progress=self.lines.append,
            **options,
        )

    def chunks(self, question: str = "loan fee policy prepayment charges process"):
        query = FakeEmbedder().embed([question])[0]
        return [chunk for chunk, _ in self.store.search_dense(query.dense, ["all"], 100)]


@pytest.fixture
def bench(tmp_cfg, store, conversions):
    return Bench(tmp_cfg, store)


# -- the normal run -------------------------------------------------------------------


def test_a_new_document_is_parsed_described_chunked_indexed_and_recorded(bench, conversions):
    bench.add("Lending Policy.pdf")

    report = bench.run()

    assert (report.files_seen, report.files_skipped, report.files_ingested) == (1, 0, 1)
    assert report.failures == ()
    assert report.figures_described == 1
    assert report.chunks_indexed == bench.store.count() > 0
    assert conversions == ["Lending Policy.pdf"]

    record = bench.db.get_document(doc_id_for("Lending Policy.pdf"))
    assert record.source_path == "Lending Policy.pdf"
    assert record.doc_title == "Lending Policy"
    assert (record.pages, record.chunks) == (2, report.chunks_indexed)

    kinds = {chunk.element_type for chunk in bench.chunks()}
    assert kinds == {ElementType.TEXT, ElementType.TABLE, ElementType.FIGURE}
    figure = next(c for c in bench.chunks() if c.element_type is ElementType.FIGURE)
    assert figure.model_generated and figure.image_path.is_file()
    assert figure.text.startswith("Figure (flowchart): Loan application process")
    assert any("parsed: 2 page(s)" in line for line in bench.lines)


def test_an_unchanged_file_is_skipped_without_parsing_or_calling_a_model(bench, conversions):
    bench.add("Lending Policy.pdf")
    first = bench.run()

    second = bench.run()

    assert (second.files_seen, second.files_skipped, second.files_ingested) == (1, 1, 0)
    assert second.chunks_indexed == 0
    assert conversions == ["Lending Policy.pdf"]
    assert len(bench.llm.requests) == 1
    assert bench.store.count() == first.chunks_indexed


def test_a_changed_file_is_parsed_again_and_replaces_its_chunks(bench, conversions):
    path = bench.add("Lending Policy.pdf")
    first = bench.run()

    path.write_bytes(b"%PDF-1.7 second version")
    second = bench.run()

    assert second.files_ingested == 1
    assert conversions == ["Lending Policy.pdf", "Lending Policy.pdf"]
    assert bench.store.count() == first.chunks_indexed
    # The picture itself did not change, so its description came from the cache.
    assert second.figures_described == 0
    assert len(bench.llm.requests) == 1


def test_a_changed_title_in_the_manifest_reindexes_without_parsing_again(bench, conversions):
    bench.add("Lending Policy.pdf")
    bench.run()

    bench.config.paths.manifest.write_text(
        "file,doc_title,effective_date\nLending Policy.pdf,Retail Lending Policy 2026,2026-04-01\n",
        encoding="utf-8",
    )
    report = bench.run()

    assert report.files_ingested == 1
    assert conversions == ["Lending Policy.pdf"]
    assert {chunk.doc_title for chunk in bench.chunks()} == {"Retail Lending Policy 2026"}
    assert {chunk.effective_date.isoformat() for chunk in bench.chunks()} == {"2026-04-01"}
    assert any("settings or metadata changed" in line for line in bench.lines)


def test_force_ingests_again_from_the_caches(bench, conversions):
    bench.add("Lending Policy.pdf")
    bench.run()

    report = bench.run(force=True)

    assert report.files_ingested == 1
    assert conversions == ["Lending Policy.pdf"]
    assert len(bench.llm.requests) == 1


def test_only_pdfs_are_taken_from_a_folder(bench):
    bench.add("Lending Policy.pdf")
    bench.add("sub/Second Policy.PDF")
    bench.add("README.md", b"# docs")
    bench.add("flow.png", b"\x89PNG")

    found = find_documents(bench.config.paths.docs_dir)

    assert [path.name for path in found] == ["Lending Policy.pdf", "Second Policy.PDF"]
    assert find_documents(bench.config.paths.docs_dir / "README.md") == []
    with pytest.raises(FileNotFoundError):
        find_documents(bench.config.paths.docs_dir / "missing")


def test_what_was_ingested_can_be_retrieved_with_its_page(bench):
    bench.add("Lending Policy.pdf")
    bench.run()

    result = retrieve(
        "What is the processing fee on a home loan?",
        bench.config,
        embedder=FakeEmbedder(),
        store=bench.store,
        reranker=FakeReranker(),
    )

    assert result.found
    # The stand-in reranker only counts shared words, so the order among the top
    # chunks means nothing here. What matters is that the table came back whole.
    table = next(item.chunk for item in result.chunks if item.chunk.element_type is ElementType.TABLE)
    assert (table.doc_title, table.page) == ("Lending Policy", 1)
    assert "Home loan | 0.50% of the loan amount | Nil" in table.text
    assert table.html.startswith("<table>")
    assert table.heading_path == ("1. Fees and charges",)


# -- figures --------------------------------------------------------------------------


def test_without_figure_descriptions_the_vision_model_is_not_called(bench):
    bench.add("Lending Policy.pdf")

    report = bench.run(describe_figures=False)

    assert report.figures_described == 0
    assert bench.llm.requests == []
    figure = next(c for c in bench.chunks() if c.element_type is ElementType.FIGURE)
    assert figure.text == "Figure caption: Figure 1: Loan application process"
    assert figure.model_generated is False

    # Switching descriptions on later counts as a change: the document is done again.
    later = bench.run()
    assert (later.files_ingested, later.figures_described) == (1, 1)


def test_a_vision_model_that_fails_fails_the_file_and_leaves_it_for_the_next_run(tmp_cfg, store, conversions):
    bench = Bench(tmp_cfg, store)
    bench.llm = ScriptedLLM(tmp_cfg, status=500)
    bench.add("Lending Policy.pdf")

    report = bench.run()

    assert report.files_ingested == 0
    assert "LLMResponseError" in report.failures[0]
    assert bench.db.get_document(doc_id_for("Lending Policy.pdf")) is None
    assert bench.store.count() == 0


# -- protection -----------------------------------------------------------------------


def test_a_file_that_looks_like_customer_data_is_not_ingested(bench):
    bench.add("customers.pdf")

    report = bench.run()

    assert report.files_ingested == 0
    assert bench.store.count() == 0
    assert bench.db.list_documents() == []
    message = report.failures[0]
    assert message.startswith("customers.pdf: not ingested")
    assert "5 PAN-like" in message and "--allow-flagged" in message
    assert "ABCPE" not in message


def test_a_flagged_file_can_be_let_through_on_purpose(bench):
    bench.add("customers.pdf")

    report = bench.run(allow_flagged=True)

    assert report.files_ingested == 1
    assert any("personal-data check flagged it" in warning for warning in report.warnings)


def test_one_file_that_fails_does_not_stop_the_others(bench):
    bench.add("a broken file.pdf")
    bench.add("Lending Policy.pdf")

    report = bench.run()

    assert (report.files_seen, report.files_ingested) == (2, 1)
    assert len(report.failures) == 1
    assert report.failures[0].startswith("a broken file.pdf: ParseError")
    assert [record.source_path for record in bench.db.list_documents()] == ["Lending Policy.pdf"]


def test_a_document_with_nothing_to_index_is_recorded_with_a_warning(bench, monkeypatch):
    from docling_core.types.doc import DoclingDocument

    monkeypatch.setattr("docqa.ingest.parser.convert_pdf", lambda path, config: DoclingDocument(name="empty"))
    bench.add("empty.pdf")

    report = bench.run()

    assert (report.files_ingested, report.chunks_indexed) == (1, 0)
    assert any("nothing was indexed" in warning for warning in report.warnings)
    assert bench.run().files_skipped == 1


def test_limit_processes_a_few_files_and_says_how_many_are_left(bench):
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        bench.add(name, name.encode())

    first = bench.run(limit=1, describe_figures=False)
    second = bench.run(limit=5, describe_figures=False)

    assert (first.files_ingested, first.files_skipped) == (1, 0)
    assert first.warnings == ("2 more file(s) still need ingesting. Run the command again to go on.",)
    assert (second.files_ingested, second.files_skipped) == (2, 1)
    assert second.warnings == ()


def test_a_large_run_stops_and_asks_first(temp_config, conversions):
    def ask_above_two(raw):
        raw["ingestion"]["confirm_above_files"] = 2

    config = temp_config(ask_above_two)
    from docqa.index.store import ChunkStore

    store = ChunkStore(config.qdrant)
    try:
        bench = Bench(config, store)
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            bench.add(name, name.encode())

        with pytest.raises(IngestAborted, match="3 files need ingesting"):
            bench.run(describe_figures=False)
        assert conversions == []
        assert bench.run(describe_figures=False, assume_yes=True).files_ingested == 3
    finally:
        store.close()


def test_prune_removes_documents_whose_files_are_gone(bench):
    keep, gone = bench.add("keep.pdf", b"keep"), bench.add("gone.pdf", b"gone")
    bench.run(describe_figures=False)
    gone_id = doc_id_for("gone.pdf")
    assert all(path.is_file() for path in cache_paths(gone_id, bench.config))

    gone.unlink()
    untouched = bench.run(describe_figures=False)
    assert bench.store.count(gone_id) > 0 and untouched.files_seen == 1

    bench.run(describe_figures=False, prune=True)

    assert bench.store.count(gone_id) == 0
    assert bench.store.count(doc_id_for("keep.pdf")) > 0
    assert [record.source_path for record in bench.db.list_documents()] == ["keep.pdf"]
    assert not any(path.exists() for path in cache_paths(gone_id, bench.config))
    assert keep.is_file()


def test_source_files_are_never_modified(bench):
    path = bench.add("Lending Policy.pdf", b"original bytes")
    before = path.stat().st_mtime_ns

    bench.run()
    bench.run(force=True)

    assert path.read_bytes() == b"original bytes"
    assert path.stat().st_mtime_ns == before


# -- command line ---------------------------------------------------------------------


def test_command_line_ingests_and_lists(bench, monkeypatch, capsys):
    bench.add("Lending Policy.pdf")
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_token_counter", lambda config: WordCounter())
    monkeypatch.setattr("docqa.resources.get_store", lambda config: bench.store)
    config_file = str(bench.config.config_file)

    assert main(["--config", config_file, "--list"]) == 0
    assert "No documents have been ingested yet." in capsys.readouterr().out

    assert main(["--config", config_file, "--no-figures"]) == 0
    output = capsys.readouterr().out
    assert "1 document(s) found, 0 unchanged, 1 to ingest." in output
    assert "Seen 1, unchanged 0, ingested 1, failed 0." in output

    assert main(["--config", config_file, "--list"]) == 0
    listing = capsys.readouterr().out
    assert "Lending Policy" in listing and "1 document(s)" in listing


def test_command_line_reports_failures_and_asks_before_a_large_run(bench, monkeypatch, capsys):
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_token_counter", lambda config: WordCounter())
    monkeypatch.setattr("docqa.resources.get_store", lambda config: bench.store)
    config_file = str(bench.config.config_file)
    bench.add("a broken file.pdf")

    assert main(["--config", config_file, "--no-figures"]) == 1
    assert "Failures:" in capsys.readouterr().out

    for number in range(30):
        bench.add(f"many/{number:02d}.pdf", str(number).encode())
    assert main(["--config", config_file, "--no-figures"]) == 2
    assert "--limit N" in capsys.readouterr().out

    assert main(["--config", str(bench.config.config_file.parent / "missing.yaml")]) == 1
    assert "not found" in capsys.readouterr().out

"""Adding documents from the chat screen: uploads, the list of waiting files, the run.

The server is driven through Starlette's test client. Docling's conversion of a PDF is
replaced by hand-built documents (tests/docling_samples.py) and the models are fakes;
the register, the index, the uploads folder and the ingestion pipeline are real.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from urllib.parse import quote

import pytest
from starlette.testclient import TestClient

from docqa.db import Database, DocumentRecord, utc_now
from docqa.ingest.documents import doc_id_for
from docqa.ingest.parser import ParseError
from docqa.tokens import WordCounter
from docqa.web import ingestion, server
from docqa.web.ingestion import IngestJob, safe_pdf_name
from docqa.web.server import STATIC_DIR, AppState, create_app
from tests.docling_samples import lending_policy, with_identifiers
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedLLM
from tests.test_web import fill, reply

PDF = b"%PDF-1.7\nfirst version of the file\n"
FIGURE_REPLY = json.dumps(
    {
        "figure_type": "flowchart",
        "title": "Loan application process",
        "nodes": ["Application received", "Credit score check", "Approve"],
        "edges": [{"source": "Application received", "target": "Credit score check", "condition": None}],
    }
)


class Screen:
    """The test client, the configuration, and a handle on the faked PDF conversion."""

    def __init__(self, client, config, converted, hold) -> None:
        self.client = client
        self.config = config
        self.converted = converted
        self.hold = hold
        """Set ``hold.clear()`` to make the conversion of a PDF wait until ``hold.set()``."""

    def put_in_docs(self, name: str, content: bytes = PDF):
        path = self.config.paths.docs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def register(self, source_path: str, content: bytes = PDF, title: str = "An Ingested Document") -> None:
        """Enter a document in the register as ingestion would, without ingesting it."""
        Database(self.config.paths.sqlite_path).save_document(
            DocumentRecord(
                doc_id_for(source_path), source_path, hashlib.sha256(content).hexdigest(), "0" * 64,
                title, None, None, "all", 3, 2, utc_now(),
            )
        )

    def upload(self, name: str, content: bytes = PDF, **headers):
        return self.client.post(f"/api/upload?name={quote(name)}", content=content, headers=headers)

    def waiting(self) -> list[str]:
        return [item["source_path"] for item in self.client.get("/api/ingest").json()["waiting"]]

    def ingest(self, files: list[str], **more):
        return self.client.post("/api/ingest", json={"files": files, "describe_figures": False, **more})

    def job(self, timeout: float = 30.0) -> dict:
        """The run once it has ended."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get("/api/ingest?files=0").json()["job"]
            if job is not None and job["state"] in ("done", "failed", "stopped"):
                return job
            time.sleep(0.02)
        raise AssertionError("The ingestion did not end.")

    def running(self, timeout: float = 10.0) -> dict:
        """The run once it has got as far as converting a file."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.converted:
                return self.client.get("/api/ingest?files=0").json()["job"]
            time.sleep(0.02)
        raise AssertionError("The ingestion did not start.")


def faked(config, monkeypatch):
    """Fake models and a fake PDF conversion. Returns the conversion's record and its hold."""
    llm = ScriptedLLM(config, reply)
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: FakeReranker())
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: llm.client)
    monkeypatch.setattr("docqa.resources.get_token_counter", lambda config: WordCounter())
    converted: list[str] = []
    hold = threading.Event()
    hold.set()

    def convert(path, config, ocr=True):
        converted.append(path.name)
        assert hold.wait(timeout=30), "the test never let the conversion go on"
        if "broken" in path.name:
            raise ParseError(f"Docling could not parse {path.name}: the file is damaged")
        if "customers" in path.name:
            return with_identifiers(5)
        return lending_policy()

    monkeypatch.setattr("docqa.ingest.parser.convert_pdf", convert)
    return converted, hold


@pytest.fixture
def screen(tmp_cfg, monkeypatch):
    """A running server over a filled index, ready to ingest."""
    fill(tmp_cfg)
    converted, hold = faked(tmp_cfg, monkeypatch)
    with TestClient(create_app(tmp_cfg, warm_up=False)) as client:
        yield Screen(client, tmp_cfg, converted, hold)
        hold.set()


# -- the page -----------------------------------------------------------------------------


def test_the_page_has_what_adding_documents_needs():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for name in ("add-docs", "ingest-dialog", "drop-zone", "file-input", "waiting-list", "ingest-start", "ingest-stop", "job-files"):
        assert f'id="{name}"' in page, name
    assert "/api/upload?name=" in script and '"/api/ingest"' in script and '"/api/ingest/stop"' in script
    assert "Public or sanitised documents only" in page


# -- what is waiting -----------------------------------------------------------------------


def test_the_pdfs_that_were_never_ingested_are_listed_as_waiting(screen):
    screen.put_in_docs("Master Directions/New Direction on Charges.pdf")
    screen.put_in_docs("Circulars/Ingested Before.pdf")
    screen.register("Circulars/Ingested Before.pdf")
    screen.put_in_docs("notes.txt", b"not a document")

    state = screen.client.get("/api/ingest").json()

    assert state["job"] is None and state["problem"] is None
    assert state["waiting"] == [
        {
            "root": "docs",
            "source_path": "Master Directions/New Direction on Charges.pdf",
            "name": "New Direction on Charges.pdf",
            "title": "New Direction on Charges",
            "short_title": "New Direction on Charges",
            "size": len(PDF),
        }
    ]
    assert state["folders"] == {
        "uploads": str(screen.config.paths.uploads_dir),
        "docs": str(screen.config.paths.docs_dir),
    }
    assert state["confirm_above"] == screen.config.ingestion.confirm_above_files
    assert state["describe_figures"] is True and state["max_upload_mb"] == screen.config.ui.max_upload_mb


def test_a_waiting_file_gets_its_title_from_the_manifest(screen):
    screen.config.paths.manifest.write_text(
        "file,doc_title,version,effective_date,access_group\n"
        "Circulars/MD42.pdf,Reserve Bank of India (Co-operative Banks - Know Your Customer) Directions,,2026-01-05,all\n",
        encoding="utf-8",
    )
    screen.put_in_docs("Circulars/MD42.pdf")
    screen.upload("MD42.pdf", PDF + b"another file of the same name")

    waiting = screen.client.get("/api/ingest").json()["waiting"]

    assert [(item["root"], item["source_path"]) for item in waiting] == [
        ("uploads", "uploads/MD42.pdf"),  # what was added here comes first
        ("docs", "Circulars/MD42.pdf"),
    ]
    assert {item["short_title"] for item in waiting} == {"Co-operative Banks - Know Your Customer, Directions"}


# -- uploads -------------------------------------------------------------------------------


def test_a_pdf_can_be_added_and_then_waits_to_be_ingested(screen):
    response = screen.upload("New Policy (2026).pdf")

    assert response.status_code == 200
    assert response.json() == {
        "status": "stored",
        "name": "New Policy (2026).pdf",
        "size": len(PDF),
        "source_path": "uploads/New Policy (2026).pdf",
    }
    stored = screen.config.paths.uploads_dir / "New Policy (2026).pdf"
    assert stored.read_bytes() == PDF
    assert screen.waiting() == ["uploads/New Policy (2026).pdf"]
    assert [path.name for path in screen.config.paths.uploads_dir.iterdir()] == ["New Policy (2026).pdf"]


@pytest.mark.parametrize(
    ("name", "content", "status", "reason"),
    [
        ("notes.txt", PDF, 400, "Only PDF files"),
        ("", PDF, 400, "Only PDF files"),
        ("report.pdf", b"PK\x03\x04 a spreadsheet with a new name", 400, "is not a PDF file"),
        ("empty.pdf", b"", 400, "is empty"),
    ],
)
def test_what_is_not_a_pdf_is_refused_and_nothing_is_left_behind(screen, name, content, status, reason):
    response = screen.upload(name, content)

    assert response.status_code == status
    assert reason in response.json()["error"]
    uploads = screen.config.paths.uploads_dir
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_a_file_over_the_size_limit_is_refused(temp_config, monkeypatch):
    def one_megabyte(raw):
        raw["ui"]["max_upload_mb"] = 1

    config = temp_config(one_megabyte)
    faked(config, monkeypatch)
    with TestClient(create_app(config, warm_up=False)) as client:
        response = client.post("/api/upload?name=big.pdf", content=PDF + b"x" * (1024 * 1024))

    assert response.status_code == 413
    assert "larger than 1 MB" in response.json()["error"]
    assert not config.paths.uploads_dir.exists() or list(config.paths.uploads_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("sent", "stored"),
    [
        ("../../evil.pdf", "evil.pdf"),
        ("C:\\Windows\\System32\\drivers.pdf", "drivers.pdf"),
        ("..pdf", None),
        ("a/b/c.PDF", "c.pdf"),
        ("Master <Direction>: 2026?.pdf", "Master _Direction__ 2026_.pdf"),
        ("मास्टर निर्देश 2026.pdf", "मास्टर निर्देश 2026.pdf"),
        ("CON.pdf", "_CON.pdf"),
        ("  spaced   out  .pdf", "spaced out.pdf"),
        ("report.pdf.exe", None),
        ("x" * 300 + ".pdf", "x" * 150 + ".pdf"),
    ],
)
def test_an_uploaded_file_is_stored_under_a_safe_name(sent, stored):
    assert safe_pdf_name(sent) == stored


def test_an_upload_cannot_be_put_outside_the_uploads_folder(screen):
    response = screen.upload("../../../config.pdf")

    assert response.json()["source_path"] == "uploads/config.pdf"
    assert (screen.config.paths.uploads_dir / "config.pdf").is_file()
    assert not (screen.config.root_dir / "config.pdf").exists()


def test_a_file_that_is_ingested_already_is_recognised_by_its_content(screen):
    screen.register("Circulars/Known.pdf", PDF, "A Known Circular")

    response = screen.upload("the same file under another name.pdf")

    assert response.json() == {
        "status": "duplicate",
        "name": "the same file under another name.pdf",
        "title": "A Known Circular",
        "source_path": "Circulars/Known.pdf",
    }
    assert list(screen.config.paths.uploads_dir.iterdir()) == []
    assert screen.waiting() == []


# -- the run -------------------------------------------------------------------------------


def test_chosen_files_are_ingested_and_then_listed_with_the_documents(screen):
    screen.put_in_docs("Circulars/Lending Policy.pdf")
    screen.upload("Recovery Policy.pdf", PDF + b"second file")
    assert screen.waiting() == ["uploads/Recovery Policy.pdf", "Circulars/Lending Policy.pdf"]

    started = screen.ingest(["Circulars/Lending Policy.pdf", "uploads/Recovery Policy.pdf"])
    job = screen.job()

    assert started.status_code == 202 and started.json()["job"]["state"] in ("waiting", "running", "done")
    assert (job["state"], job["what"], job["total"], job["done"], job["failed"]) == ("done", "chosen", 2, 2, 0)
    assert job["error"] is None and job["current"] is None
    assert [(file["source_path"], file["state"]) for file in job["files"]] == [
        ("Circulars/Lending Policy.pdf", "done"),
        ("uploads/Recovery Policy.pdf", "done"),
    ]
    assert all("passage(s), 2 page(s)" in file["detail"] for file in job["files"])
    report = job["report"]
    assert (report["files_ingested"], report["files_to_ingest"], report["failures"], report["stopped"]) == (2, 2, [], False)
    assert report["chunks_indexed"] > 0 and report["figures_described"] == 0
    assert sorted(screen.converted) == ["Lending Policy.pdf", "Recovery Policy.pdf"]
    assert any("indexed uploads/Recovery Policy.pdf" in line for line in job["lines"])

    documents = screen.client.get("/api/documents").json()["documents"]
    assert {doc["source_path"] for doc in documents} >= {"Circulars/Lending Policy.pdf", "uploads/Recovery Policy.pdf"}
    assert screen.waiting() == []
    status = screen.client.get("/api/status").json()
    assert status["index"]["documents"] == 5 and status["ingesting"] is False
    # The new documents can be searched in the same server, without a restart.
    stream = screen.client.post("/api/ask", json={"question": "home loan processing fee"}).text
    assert "event: answer" in stream


def test_the_page_can_ask_only_for_the_lines_it_has_not_seen(screen):
    screen.put_in_docs("Lending Policy.pdf")
    screen.ingest(["Lending Policy.pdf"])
    job = screen.job()

    later = screen.client.get(f"/api/ingest?files=0&since={job['line_count'] - 1}").json()

    assert "waiting" not in later  # the list of files is left out when the page only wants the run
    assert later["job"]["lines"] == job["lines"][-1:] and later["job"]["line_count"] == job["line_count"]


def test_figures_are_described_when_the_box_is_ticked(screen, monkeypatch):
    vision = ScriptedLLM(screen.config, FIGURE_REPLY)
    monkeypatch.setattr("docqa.ingest.pipeline.LLMClient", lambda settings: vision.client)
    screen.put_in_docs("Lending Policy.pdf")

    screen.client.post("/api/ingest", json={"files": ["Lending Policy.pdf"], "describe_figures": True})
    job = screen.job()

    assert job["state"] == "done" and job["describe_figures"] is True
    assert job["report"]["figures_described"] == 1 and len(vision.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"files": []},
        {"files": "Lending Policy.pdf"},
        {"files": [7]},
        {},
        {"files": ["../config.yaml"]},
        {"files": ["Ingested Before.pdf"]},  # ingested already, so it is not waiting
        {"files": ["Lending Policy.pdf", "C:/Windows/win.ini"]},
    ],
)
def test_only_files_that_are_waiting_can_be_ingested(screen, body):
    screen.put_in_docs("Lending Policy.pdf")
    screen.put_in_docs("Ingested Before.pdf")
    screen.register("Ingested Before.pdf")

    response = screen.client.post("/api/ingest", json=body)

    assert response.status_code == 400
    assert screen.client.get("/api/ingest?files=0").json()["job"] is None
    assert screen.converted == []


def test_a_request_that_is_not_json_is_refused(screen):
    assert screen.client.post("/api/ingest", content="files=all").status_code == 400
    assert screen.client.post("/api/ingest", json=["Lending Policy.pdf"]).status_code == 400


def test_a_large_run_asks_before_it_starts(temp_config, monkeypatch):
    def ask_above_one(raw):
        raw["ingestion"]["confirm_above_files"] = 1

    config = temp_config(ask_above_one)
    converted, _hold = faked(config, monkeypatch)
    with TestClient(create_app(config, warm_up=False)) as client:
        screen = Screen(client, config, converted, _hold)
        screen.put_in_docs("a.pdf")
        screen.put_in_docs("b.pdf", PDF + b"b")

        asked = screen.ingest(["a.pdf", "b.pdf"])
        assert asked.status_code == 409
        assert asked.json()["needs_confirmation"] is True and asked.json()["files"] == 2
        assert "more than 1" in asked.json()["error"] and converted == []

        confirmed = screen.ingest(["a.pdf", "b.pdf"], confirmed=True)
        assert confirmed.status_code == 202
        assert screen.job()["done"] == 2


def test_one_ingestion_runs_at_a_time_and_nothing_is_added_meanwhile(screen):
    screen.put_in_docs("a.pdf")
    screen.put_in_docs("b.pdf", PDF + b"b")
    screen.hold.clear()  # the conversion of the first file waits

    assert screen.ingest(["a.pdf"]).status_code == 202
    running = screen.running()
    second = screen.ingest(["b.pdf"])
    upload = screen.upload("c.pdf")
    status = screen.client.get("/api/status").json()
    screen.hold.set()
    finished = screen.job()

    assert running["state"] == "running" and running["current"]["name"] == "a.pdf"
    assert (running["current"]["step"], running["current"]["position"], running["current"]["of"]) == ("read", 1, 1)
    assert running["files"][0]["state"] == "reading"
    assert second.status_code == 409 and "already running" in second.json()["error"]
    assert upload.status_code == 409 and "being ingested" in upload.json()["error"]
    assert status["ingesting"] is True
    assert finished["state"] == "done" and screen.waiting() == ["b.pdf"]


def test_an_ingestion_can_be_stopped(screen):
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        screen.put_in_docs(name, PDF + name.encode())
    screen.hold.clear()

    screen.ingest(["a.pdf", "b.pdf", "c.pdf"])
    screen.running()
    stopping = screen.client.post("/api/ingest/stop")
    screen.hold.set()
    job = screen.job()

    assert stopping.status_code == 200 and stopping.json()["job"]["state"] == "stopping"
    assert job["state"] == "stopped" and job["report"]["stopped"] is True
    assert screen.converted == ["a.pdf"]  # the file in hand was finished, the next ones were not begun
    assert job["report"]["files_ingested"] == 0 and job["done"] == 0
    assert [file["state"] for file in job["files"]] == ["left", "left", "left"]  # none of them got through
    assert any(line.startswith("Stopped.") for line in job["lines"])
    assert screen.waiting() == ["a.pdf", "b.pdf", "c.pdf"]  # all three can be ingested later
    assert screen.client.post("/api/ingest/stop").status_code == 409  # nothing is running now


def test_a_file_that_looks_like_customer_data_is_not_ingested_and_the_others_are(screen):
    screen.put_in_docs("customers.pdf")
    screen.put_in_docs("Lending Policy.pdf", PDF + b"policy")
    screen.put_in_docs("broken.pdf", PDF + b"broken")

    screen.ingest(["customers.pdf", "Lending Policy.pdf", "broken.pdf"])
    job = screen.job()

    states = {file["name"]: file for file in job["files"]}
    assert (job["state"], job["done"], job["failed"]) == ("done", 1, 2)
    assert states["customers.pdf"]["state"] == "failed"
    assert "personal data" in states["customers.pdf"]["detail"]
    assert states["broken.pdf"]["state"] == "failed" and "damaged" in states["broken.pdf"]["detail"]
    assert states["Lending Policy.pdf"]["state"] == "done"
    assert len(job["report"]["failures"]) == 2
    assert screen.waiting() == ["broken.pdf", "customers.pdf"]


def test_ingested_documents_can_be_brought_up_to_date_without_taking_in_new_files(screen):
    policy = screen.put_in_docs("Lending Policy.pdf")
    screen.ingest(["Lending Policy.pdf"])
    screen.job()
    policy.write_bytes(PDF + b"a second version")
    screen.put_in_docs("Never Ingested.pdf", PDF + b"new")
    screen.converted.clear()

    started = screen.client.post("/api/ingest", json={"refresh": True, "describe_figures": False})
    job = screen.job()

    assert started.status_code == 202
    assert (job["state"], job["what"]) == ("done", "refresh")
    assert [(file["source_path"], file["state"]) for file in job["files"]] == [("Lending Policy.pdf", "done")]
    assert job["report"]["files_ingested"] == 1
    assert screen.converted == ["Lending Policy.pdf"]
    assert screen.waiting() == ["Never Ingested.pdf"]


def test_a_question_waits_while_documents_are_being_ingested(screen):
    state = AppState(screen.config, warm_up=False)
    state.job = IngestJob(1, False, "chosen", ["a.pdf"])
    state.busy.acquire()  # as the run holds it
    seen: list[tuple] = []

    worker = threading.Thread(
        target=server._answer_question,
        args=(state, "home loan processing fee", lambda event, data: seen.append((event, data.get("name"), data.get("reason"))), threading.Event()),
        daemon=True,
    )
    worker.start()
    for _ in range(200):
        if seen:
            break
        time.sleep(0.01)
    state.job.finish("done", ingestion.merged([]))
    state.busy.release()
    worker.join(timeout=30)

    assert seen[0] == ("step", "waiting", "ingestion")
    assert not worker.is_alive() and seen[-1][0] == "answer"


def test_the_search_models_make_room_for_the_run_and_are_loaded_again_after_it(tmp_cfg, monkeypatch):
    released: list[str] = []
    monkeypatch.setattr("docqa.resources.release", released.append)
    monkeypatch.setattr("docqa.resources.warm_up", lambda config: {"store": 0.1, "embedder": 9.0, "reranker": 8.0})
    state = AppState(tmp_cfg, warm_up=True)
    job = IngestJob(1, False, "chosen", [])
    state.job = job

    ingestion.run(state, job, [])

    assert released == ["reranker", "embedder"]
    assert state.models["state"] == "ready" and state.models["seconds"]["embedder"] == 9.0
    assert job.state == "done" and not job.active
    assert job.snapshot()["lines"] == ["Loading the search models again, for questions."]
    assert state.busy.acquire(blocking=False)  # the lock was given back
    state.busy.release()


def test_a_run_that_was_stopped_before_it_began_does_nothing(tmp_cfg):
    state = AppState(tmp_cfg, warm_up=False)
    job = IngestJob(1, False, "chosen", ["a.pdf"])
    job.ask_to_stop()

    ingestion.run(state, job, [{"folder": tmp_cfg.paths.docs_dir, "source_root": tmp_cfg.paths.docs_dir, "only": ["a.pdf"], "known_only": False}])

    assert job.state == "stopped" and job.snapshot()["report"]["files_ingested"] == 0


# -- the run, as the page is told it --------------------------------------------------------


def test_the_run_reports_each_file_step_by_step():
    job = IngestJob(3, True, "chosen", ["a.pdf", "Circulars/b.pdf"])
    assert job.active and [file["state"] for file in job.snapshot()["files"]] == ["waiting", "waiting"]

    job.begin()
    job.say("2 document(s) found, 0 unchanged, 2 to ingest.")
    job.on_event("plan", {"seen": 2, "unchanged": 0, "left_alone": 0, "files": ["a.pdf", "Circulars/b.pdf"]})
    job.on_event("file", {"step": "read", "source_path": "a.pdf", "position": 1, "of": 2})
    first = job.snapshot()
    job.on_event("file", {"step": "read", "source_path": "Circulars/b.pdf", "position": 2, "of": 2})
    job.on_event("file", {"step": "figures", "source_path": "a.pdf", "position": 1, "of": 2})
    job.on_event("figure", {"source_path": "a.pdf", "position": 2, "of": 3, "page": 7})
    figures = job.snapshot()
    job.on_event("file", {"step": "index", "source_path": "a.pdf", "position": 1, "of": 2})
    job.on_event("file_done", {"source_path": "a.pdf", "chunks": 12, "pages": 4, "notes": ["a long table"]})
    job.on_event("file", {"step": "index", "source_path": "Circulars/b.pdf", "position": 2, "of": 2})
    job.on_event("file_failed", {"source_path": "Circulars/b.pdf", "problem": "ParseError: damaged"})
    job.finish("done", ingestion.merged([]))
    last = job.snapshot(since=1)

    assert first["state"] == "running" and first["current"] == {
        "source_path": "a.pdf", "name": "a.pdf", "step": "read", "position": 1, "of": 2,
    }
    assert [file["state"] for file in first["files"]] == ["reading", "waiting"]
    # While its figures are described, the other file waits for its next step.
    assert [file["state"] for file in figures["files"]] == ["figures", "between"]
    assert figures["current"]["figure"] == {"position": 2, "of": 3, "page": 7}
    assert [(file["name"], file["state"], file["detail"]) for file in last["files"]] == [
        ("a.pdf", "done", "12 passage(s), 4 page(s)"),
        ("b.pdf", "failed", "ParseError: damaged"),
    ]
    assert last["files"][0]["notes"] == ["a long table"]
    assert (last["state"], last["done"], last["failed"], last["total"], last["current"]) == ("done", 1, 1, 2, None)
    assert last["lines"] == [] and last["line_count"] == 1 and not job.active


def test_a_refresh_learns_its_files_from_the_run():
    job = IngestJob(1, False, "refresh", [])

    job.on_event("plan", {"seen": 600, "unchanged": 13, "left_alone": 585, "files": ["x.pdf", "y.pdf"]})

    assert [file["source_path"] for file in job.snapshot()["files"]] == ["x.pdf", "y.pdf"]


def test_a_long_run_keeps_only_its_latest_lines():
    job = IngestJob(1, False, "chosen", [])
    for number in range(2500):
        job.say(f"line {number}")

    everything = job.snapshot(since=0)
    recent = job.snapshot(since=2498)

    assert everything["line_count"] == 2500 and len(everything["lines"]) == 2000
    assert everything["lines"][0] == "line 500"
    assert recent["lines"] == ["line 2498", "line 2499"]


# -- requests from elsewhere ---------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site", "Origin": "http://testserver"},
    ],
)
def test_another_website_cannot_make_the_browser_start_anything(screen, headers):
    screen.put_in_docs("Lending Policy.pdf")

    refused = [
        screen.client.post("/api/ask", json={"question": "home loan processing fee"}, headers=headers),
        screen.upload("x.pdf", **headers),
        screen.client.post("/api/ingest", json={"files": ["Lending Policy.pdf"]}, headers=headers),
        screen.client.post("/api/ingest/stop", headers=headers),
    ]

    assert [response.status_code for response in refused] == [403, 403, 403, 403]
    assert "did not come from the chat screen" in refused[0].json()["error"]
    assert screen.converted == [] and screen.waiting() == ["Lending Policy.pdf"]


def test_the_page_itself_and_programs_that_are_not_browsers_are_let_through(screen):
    own = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}

    assert screen.upload("x.pdf", **own).status_code == 200
    assert screen.upload("y.pdf", PDF + b"y").status_code == 200  # no such headers at all
    assert screen.client.post("/api/ask", json={"question": "  "}, headers=own).status_code == 400

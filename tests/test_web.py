"""The chat screen's server, through Starlette's own test client (no browser, no port).

The search models and the answer model are fakes. The index is a real Qdrant store in
a temporary folder, filled before the server is started.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import date

import pytest
from PIL import Image
from starlette.testclient import TestClient

from docqa.config import ConfigError
from docqa.db import Database, DocumentRecord, utc_now
from docqa.generate import build_messages
from docqa.index import index_chunks
from docqa.index.store import ChunkStore, StoreError
from docqa.schema import ElementType, RetrievalResult, RetrievalTimings, ScoredChunk
from docqa.tokens import WordCounter
from docqa.web import ingestion, server
from docqa.web.ingestion import IngestJob, safe_pdf_name
from docqa.web.server import STATIC_DIR, AppState, Cancelled, create_app
from tests.docling_samples import lending_policy, with_identifiers
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedLLM, make_chunk

NOT_FOUND = "I could not find this in the documents."
FIGURE_DOC = "0123456789abcdef"


def corpus(config):
    figure = config.paths.figures_dir / FIGURE_DOC / "p0007-f001.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 40), "white").save(figure)
    return [
        make_chunk(
            "fees:00000",
            "Table: Schedule of charges\nColumns: Loan type | Processing fee\nHome loan | 0.50% of the loan amount",
            doc_title="Reserve Bank of India (Commercial Banks - Schedule of Charges) Directions, 2026",
            page=12,
            element_type=ElementType.TABLE,
            effective_date=date(2026, 4, 1),
            html="<table><tr><th>Loan type</th><th>Processing fee</th></tr>"
            "<tr><td>Home loan</td><td>0.50% of the loan amount</td></tr></table>",
        ),
        make_chunk(
            f"{FIGURE_DOC}:00000",
            "Figure (flowchart): Home loan processing fee waiver process\nArrows:\n- Request -> Approval",
            doc_title="Credit Process Manual",
            page=7,
            element_type=ElementType.FIGURE,
            image_path=figure,
            model_generated=True,
        ),
        make_chunk("hours:00000", "Branches open at ten and close at five.", doc_title="Branch Manual", page=2),
    ]


def reply(body: dict) -> str:
    return "The processing fee on a home loan is 0.50% of the loan amount [E1, p. 12]."


def events(response) -> list[tuple[str, dict]]:
    """The server-sent events of a finished response, in order."""
    found = []
    for block in response.text.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if name:
            found.append((name, data))
    return found


def one(stream: list[tuple[str, dict]], wanted: str) -> dict:
    """The data of the first event called ``wanted``."""
    return next(data for name, data in stream if name == wanted)


def steps(stream: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    """The steps of an answer in the order the page was told about them."""
    return [(data["name"], data["state"]) for name, data in stream if name == "step"]


def fill(config) -> None:
    """Index the corpus and register its documents, then close the store for the server."""
    store = ChunkStore(config.qdrant)
    try:
        index_chunks(corpus(config), config, embedder=FakeEmbedder(), store=store)
    finally:
        store.close()
    db = Database(config.paths.sqlite_path)
    for doc_id, path, title, pages, when in (
        ("fees", "Master Directions/Charges.pdf", "Reserve Bank of India (Commercial Banks - Schedule of Charges) Directions, 2026", 30, date(2026, 4, 1)),
        (FIGURE_DOC, "Credit Process Manual.pdf", "Credit Process Manual", 12, None),
        ("hours", "Branch Manual.pdf", "Branch Manual", 4, None),
    ):
        db.save_document(DocumentRecord(doc_id, path, "0" * 64, "0" * 64, title, None, when, "all", pages, 1, utc_now()))


@pytest.fixture
def ready(tmp_cfg, monkeypatch):
    """A running server over a filled index, with fake models. Yields (client, llm, config)."""
    fill(tmp_cfg)
    llm = ScriptedLLM(tmp_cfg, reply)
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: FakeReranker())
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: llm.client)
    with TestClient(create_app(tmp_cfg, warm_up=False)) as client:
        yield client, llm, tmp_cfg


# -- the page ----------------------------------------------------------------------------


def test_the_page_and_its_files_are_served(ready):
    client, _llm, _config = ready

    page = client.get("/")

    assert page.status_code == 200
    assert "<title>BFSI document Q&amp;A</title>" in page.text
    assert 'id="doc-list"' in page.text and 'id="tiles"' in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/missing.js").status_code == 404


def test_the_page_fetches_nothing_from_outside_this_machine():
    """Nothing leaves the machine: no fonts, scripts or images from the web."""
    for name in ("index.html", "app.css", "app.js"):
        text = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://|//cdn|@import", text), name


# -- status, documents, stats ------------------------------------------------------------


def test_status_says_ready_when_everything_is_in_place(ready):
    client, _llm, config = ready

    status = client.get("/api/status").json()

    assert status["ready"] is True and status["problems"] == []
    assert status["index"] == {"documents": 3, "passages": 3}
    assert status["threshold"] == config.retrieval.rerank_score_threshold
    assert status["llm"] == {"reachable": True, "answer_model_present": True, "detail": ""}
    assert status["settings"]["answer_model"] == "qwen2.5:7b"
    assert status["settings"]["answer_model_host"] is None  # on this machine
    assert status["settings"]["candidates"] == 15
    assert status["settings"]["reranker_8_bit"] is True
    assert "time_limit_s" not in status["settings"]  # an answer takes as long as it takes
    assert status["ingesting"] is False


def test_status_says_when_the_answer_model_is_on_another_machine(temp_config, monkeypatch):
    """The page then shows where every question and its passages are sent."""

    def hosted(raw):
        raw["llm"].update(base_url="https://models.example.com/v1", allow_remote=True)

    config = temp_config(hosted)
    llm = ScriptedLLM(config, reply)
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: llm.client)

    with TestClient(create_app(config, warm_up=False)) as client:
        status = client.get("/api/status").json()

    assert status["settings"]["answer_model_host"] == "models.example.com"
    assert "answer_model_host" in (STATIC_DIR / "app.js").read_text(encoding="utf-8")


def test_status_names_what_is_missing(temp_config, monkeypatch):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    config = temp_config(unset)
    down = ScriptedLLM(config, status=500)
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: down.client)

    with TestClient(create_app(config, warm_up=False)) as client:
        status = client.get("/api/status").json()

    assert status["ready"] is False
    problems = " ".join(status["problems"])
    assert "No documents are indexed yet" in problems
    assert "rerank_score_threshold is not set" in problems
    assert "The answer model cannot be used" in problems
    assert status["llm"]["reachable"] is False


def test_documents_come_with_a_short_name_and_their_full_title(ready):
    client, _llm, _config = ready

    documents = client.get("/api/documents").json()["documents"]

    assert [doc["short_title"] for doc in documents] == [
        "Branch Manual",
        "Commercial Banks - Schedule of Charges, Directions, 2026",
        "Credit Process Manual",
    ]
    charges = documents[1]
    assert charges["title"] == "Reserve Bank of India (Commercial Banks - Schedule of Charges) Directions, 2026"
    assert (charges["pages"], charges["passages"], charges["effective_date"]) == (30, 1, "2026-04-01")
    assert charges["source_path"] == "Master Directions/Charges.pdf"


def test_stats_cover_the_index_and_the_questions_asked(ready):
    client, _llm, _config = ready

    before = client.get("/api/stats").json()
    client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"})
    client.post("/api/ask", json={"question": "Which planets have rings?"})
    after = client.get("/api/stats").json()

    assert before["index"]["documents"] == 3 and before["index"]["pages"] == 46
    assert (before["index"]["text_passages"], before["index"]["tables"], before["index"]["figures"]) == (1, 1, 1)
    assert before["queries"]["questions"] == 0 and before["queries"]["median_latency_s"] is None
    queries = after["queries"]
    assert (queries["questions"], queries["answered"], queries["refused"], queries["failed"]) == (2, 1, 1, 0)
    assert queries["median_latency_s"] > 0 and queries["median_search_s"] > 0
    assert queries["median_first_token_s"] is not None
    assert queries["last_asked_at"] is not None


def test_history_lists_the_latest_questions_first(ready):
    client, _llm, _config = ready
    client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"})
    client.post("/api/ask", json={"question": "Which planets have rings?"})

    questions = client.get("/api/history?limit=5").json()["questions"]

    assert [item["question"] for item in questions] == [
        "Which planets have rings?",
        "What is the processing fee on a home loan?",
    ]
    assert questions[0]["refused"] is True and questions[1]["refused"] is False


# -- asking ------------------------------------------------------------------------------


def test_retrieval_payload_hides_sources_below_the_rerank_threshold(temp_config):
    def threshold(raw):
        raw["retrieval"]["rerank_score_threshold"] = 0.5

    config = temp_config(threshold)
    above = ScoredChunk(make_chunk("above", "Relevant text"), 0.8, 0.75)
    below = ScoredChunk(make_chunk("below", "Weak text"), 0.4, 0.49)
    retrieval = RetrievalResult(
        "question",
        (above, below),
        True,
        0.75,
        timings=RetrievalTimings(candidates=2),
    )

    payload = server._retrieval_payload(retrieval, (above,), config)

    assert [source["chunk_id"] for source in payload["sources"]] == ["above"]


def test_a_question_is_answered_as_a_stream_of_steps(ready):
    client, llm, config = ready

    response = client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    names = [name for name, _ in stream]
    assert names[0] == "start" and stream[0][1] == {}
    assert names[-2:] == ["answer", "done"]
    assert set(names[1:-2]) == {"step", "retrieval", "token"}
    # Every step is announced when it starts and when it ends, in the order it happens.
    assert steps(stream) == [
        ("question", "start"),
        ("question", "end"),
        ("database", "start"),
        ("database", "end"),
        ("ranking", "start"),
        ("ranking", "progress"),
        ("ranking", "end"),
        ("reading", "start"),
        ("reading", "end"),
        ("writing", "start"),
        ("writing", "end"),
    ]
    # The passages reach the page after the ranking and before the model starts reading.
    order = [name if name != "step" else f"{data['name']} {data['state']}" for name, data in stream]
    assert order.index("ranking end") < order.index("retrieval") < order.index("reading start")
    assert order.index("writing start") < order.index("token")
    ranking = [data for name, data in stream if name == "step" and data["name"] == "ranking"]
    assert ranking[0]["of"] == 3 and ranking[1]["done"] == 3 and ranking[2]["took_s"] >= 0
    reading = next(data for name, data in stream if name == "step" and data["name"] == "reading")
    assert reading["passages"] >= 1 and reading["chars"] > 50

    retrieval = one(stream, "retrieval")
    assert retrieval["expected_read_s"] is None  # nothing has been answered here yet
    assert retrieval["found"] is True and retrieval["threshold"] == config.retrieval.rerank_score_threshold
    assert retrieval["timings"]["candidates"] == 3
    first = retrieval["sources"][0]
    assert (first["rank"], first["chunk_id"], first["type"], first["pages"]) == (1, "fees:00000", "table", "p. 12")
    assert first["short_title"] == "Commercial Banks - Schedule of Charges, Directions, 2026"
    assert first["html"].startswith("<table>") and first["given_to_model"] is True
    assert first["effective_date"] == "2026-04-01" and first["image_url"] is None

    # The page shows the model's own words while they arrive ...
    assert "".join(data["text"] for name, data in stream if name == "token").startswith(
        "The processing fee on a home loan is 0.50% of the loan amount [E1, p. 12]."
    )
    # ... and gets the checked answer at the end, with the citation written out.
    final = stream[-2][1]
    title = "Reserve Bank of India (Commercial Banks - Schedule of Charges) Directions, 2026"
    assert final["text"] == f"The processing fee on a home loan is 0.50% of the loan amount [{title}, p. 12]."
    assert final["refused"] is False and final["notes"] == [] and final["unmatched"] == []
    assert final["citations"] == [
        {
            "doc_title": title,
            "short_title": "Commercial Banks - Schedule of Charges, Directions, 2026",
            "page": 12,
            "chunk_id": "fees:00000",
            "rank": 1,
            "label": f"[{title}, p. 12]",
        }
    ]
    assert final["answer_model"] == "qwen2.5:7b" and final["latency_s"] > 0
    assert final["timings"]["retrieval"]["candidates"] == 3
    assert final["timings"]["first_token_s"] is not None
    assert final["evidence_ids"][0] == "fees:00000"
    assert len(llm.requests) == 1 and llm.requests[0]["stream"] is True

    logged = Database(config.paths.sqlite_path).recent_queries()[0]
    assert logged.question == "What is the processing fee on a home loan?"
    assert logged.answer == final["text"]
    assert logged.timings["retrieval"]["candidates"] == 3


def test_the_page_is_told_how_long_the_reading_can_be_expected_to_take(ready):
    """From the second answer on: the first one is what the model's speed is measured from."""
    client, _llm, config = ready
    question = {"question": "What is the processing fee on a home loan?"}

    first = events(client.post("/api/ask", json=question))
    second = events(client.post("/api/ask", json=question))

    logged = Database(config.paths.sqlite_path).recent_queries()[-1].timings
    speed = logged["evidence_chars"] / logged["first_token_s"]
    reading = next(data for name, data in second if name == "step" and data["name"] == "reading")
    assert one(first, "retrieval")["expected_read_s"] is None
    assert one(second, "retrieval")["expected_read_s"] == pytest.approx(reading["chars"] / speed, rel=0.01)
    # It is an estimate for the reader. The model is given the same evidence either way.
    assert one(first, "answer")["evidence_ids"] == one(second, "answer")["evidence_ids"]


def test_sources_say_which_passages_went_to_the_model(ready):
    client, _llm, _config = ready

    stream = events(client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"}))

    sources = one(stream, "retrieval")["sources"]
    given = [source["chunk_id"] for source in sources if source["given_to_model"]]
    assert given and given[0] == "fees:00000"
    assert stream[-2][1]["evidence_ids"] == given
    # The opening hours share no word with the question, so that passage is not sent.
    hours = next((source for source in sources if source["chunk_id"] == "hours:00000"), None)
    assert hours is None or hours["given_to_model"] is False


def test_a_question_the_documents_cannot_answer_is_refused_without_the_model(ready):
    client, llm, config = ready

    stream = events(client.post("/api/ask", json={"question": "Which planets have rings?"}))

    retrieval, final = one(stream, "retrieval"), stream[-2][1]
    assert retrieval["found"] is False
    assert not [name for name, _state in steps(stream) if name in ("reading", "writing")]  # no model, no such steps
    assert all(source["given_to_model"] is False for source in retrieval["sources"])
    assert final["refused"] is True and final["text"] == NOT_FOUND
    assert final["answer_model"] is None and final["citations"] == []
    assert llm.requests == []
    assert Database(config.paths.sqlite_path).recent_queries()[0].refused is True


def test_a_figure_source_carries_an_address_for_its_picture(ready):
    client, _llm, config = ready

    stream = events(client.post("/api/ask", json={"question": "home loan processing fee waiver process"}))
    figure = next(source for source in one(stream, "retrieval")["sources"] if source["type"] == "figure")

    assert figure["image_url"] == f"/api/figure/{FIGURE_DOC}/p0007-f001.png"
    assert figure["model_generated"] is True
    picture = client.get(figure["image_url"])
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/png"
    assert picture.content == (config.paths.figures_dir / FIGURE_DOC / "p0007-f001.png").read_bytes()


@pytest.mark.parametrize(
    "address",
    [
        f"/api/figure/{FIGURE_DOC}/missing.png",
        "/api/figure/not-a-doc-id/p0007-f001.png",
        f"/api/figure/{FIGURE_DOC}/..%2F..%2Fdocqa.sqlite",
        f"/api/figure/{FIGURE_DOC}/secret.txt",
    ],
)
def test_only_figure_images_can_be_fetched(ready, address):
    client, _llm, _config = ready
    assert client.get(address).status_code == 404


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"question": "   "}, "empty"),
        ({"question": "x" * 2001}, "longer than 2000"),
        ({"query": "wrong key"}, "'question'"),
        ("not json", "'question'"),
    ],
)
def test_a_bad_request_is_refused_with_a_reason(ready, body, expected):
    client, llm, _config = ready

    response = client.post("/api/ask", content=body) if isinstance(body, str) else client.post("/api/ask", json=body)

    assert response.status_code == 400
    assert expected in response.json()["error"]
    assert llm.requests == []


def test_a_failing_model_server_is_reported_and_logged(ready, monkeypatch):
    client, _llm, config = ready
    broken = ScriptedLLM(config, status=500)
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: broken.client)

    stream = events(client.post("/api/ask", json={"question": "What is the processing fee on a home loan?"}))

    names = [name for name, _ in stream]
    assert names[-2:] == ["error", "done"] and "answer" not in names
    assert stream[-2][1]["message"].startswith("LLMResponseError")
    logged = Database(config.paths.sqlite_path).recent_queries()[0]
    assert logged.error.startswith("LLMResponseError") and logged.answer == ""


def test_asking_before_the_threshold_is_set_explains_what_to_do(temp_config, monkeypatch):
    def unset(raw):
        raw["retrieval"]["rerank_score_threshold"] = None

    config = temp_config(unset)
    fill(config)
    monkeypatch.setattr("docqa.resources.get_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr("docqa.resources.get_reranker", lambda config: FakeReranker())

    with TestClient(create_app(config, warm_up=False)) as client:
        stream = events(client.post("/api/ask", json={"question": "Any question"}))

    assert [name for name, _ in stream][-2:] == ["error", "done"]
    assert "--suggest-threshold" in stream[-2][1]["message"]


# -- stopping, queueing, warming up ---------------------------------------------------------


def test_an_answer_is_dropped_when_the_page_stops_waiting(ready):
    """Stopping must not leave a logged answer behind, and must free the server for the next question."""
    _client, llm, config = ready
    state = AppState(config, warm_up=False)
    cancel = threading.Event()
    seen: list[str] = []

    def emit(event: str, data: dict) -> None:
        seen.append(event)
        if event == "token":
            cancel.set()  # the page goes away after the first word

    with pytest.raises(Cancelled):
        server._answer_question(state, "What is the processing fee on a home loan?", emit, cancel)

    assert seen.count("token") == 1 and "answer" not in seen
    assert Database(config.paths.sqlite_path).recent_queries() == []
    assert state.busy.acquire(blocking=False)  # the lock was given back
    state.busy.release()
    assert len(llm.requests) == 1


def test_a_second_question_waits_for_the_first(ready):
    _client, _llm, config = ready
    state = AppState(config, warm_up=False)
    seen: list[str] = []
    state.busy.acquire()  # as if another question were being answered

    worker = threading.Thread(
        target=server._answer_question,
        args=(
            state,
            "What is the processing fee on a home loan?",
            lambda event, data: seen.append((event, data.get("name"), data.get("state"), data.get("reason"))),
            threading.Event(),
        ),
        daemon=True,  # a failure below must not leave the test run waiting for this thread
    )
    worker.start()
    for _ in range(200):
        if seen:
            break
        threading.Event().wait(0.01)
    assert seen == [("step", "waiting", "start", "question")]

    state.busy.release()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert seen[1] == ("step", "waiting", "end", None) and seen[2][0] == "start"
    assert seen[-1][0] == "answer"


def test_warming_up_loads_the_models_and_wakes_the_answer_model(tmp_cfg, monkeypatch):
    llm = ScriptedLLM(tmp_cfg, "OK")
    monkeypatch.setattr("docqa.resources.warm_up", lambda config: {"store": 0.1, "embedder": 20.0, "reranker": 15.0})
    monkeypatch.setattr("docqa.resources.get_llm", lambda config: llm.client)
    state = AppState(tmp_cfg, warm_up=True)
    assert state.models["state"] == "loading"

    state.warm_up()

    assert state.models["state"] == "ready"
    assert state.models["seconds"]["embedder"] == 20.0 and "answer_model" in state.models["seconds"]
    assert llm.requests[0]["max_tokens"] == 1
    # The model is woken with the instructions of a real question in front, so that it
    # has read them already when the first question comes.
    wake, question = llm.requests[0]["messages"], build_messages("Any question?", [], tmp_cfg)
    assert wake[0] == question[0] and wake[0]["role"] == "system"


def test_a_model_that_cannot_be_loaded_is_shown_not_raised(tmp_cfg, monkeypatch):
    def fail(config):
        raise StoreError("The index is in use by another program.")

    monkeypatch.setattr("docqa.resources.warm_up", fail)
    state = AppState(tmp_cfg, warm_up=True)

    state.warm_up()

    assert state.models["state"] == "failed"
    assert "in use by another program" in state.models["detail"]
    with TestClient(create_app(tmp_cfg, warm_up=False)) as client:
        client.app.state.docqa.models = state.models
        problems = client.get("/api/status").json()["problems"]
    assert any("search models could not be loaded" in problem for problem in problems)


# -- where it listens -------------------------------------------------------------------------


def test_the_screen_listens_on_this_machine_only_unless_told_otherwise(config, temp_config):
    assert config.ui.host == "127.0.0.1" and config.ui.allow_network is False

    def open_to_network(raw):
        raw["ui"]["host"] = "0.0.0.0"

    with pytest.raises(ConfigError, match="reachable from the network"):
        temp_config(open_to_network)

    def on_purpose(raw):
        raw["ui"]["host"] = "0.0.0.0"
        raw["ui"]["allow_network"] = True

    assert temp_config(on_purpose).ui.host == "0.0.0.0"


def test_command_line_reports_a_missing_configuration(tmp_path, capsys):
    assert server.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "not found" in capsys.readouterr().out

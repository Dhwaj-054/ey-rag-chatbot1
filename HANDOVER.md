# Handover: setting up, updating and speeding up the build

Written on 2026-11-20. Updated on 2026-11-23 with release 2026-11-23.

Read this before running anything. It says what exists, what has and has not been
checked, where the time of an answer goes and what to try about it, how to update a
copy that is already set up, how to set up a new machine, and where the code is most
likely to need fixing. `CLAUDE.md` holds the rules of the project, `design.md` explains
the methods, and `README.md` lists the commands.

## Where things stand

All of the pipeline code is written: ingestion, indexing, retrieval, cited answers, the
chat screen, the query log, the statistics and the evaluation script.

Two laptops are involved (`machine.md` describes both):

- **Laptop B, where the code is written,** has no Ollama program, no BGE-M3, no
  reranker and none of Docling's models. Only the fast tests run there, with small
  stand-ins in place of the models.
- **Laptop A has the models.** Release 2026-11-20 was set up there with GitHub Copilot.
  On 2026-11-21 it ingested 15 PDFs in 1,573 s (about 105 s each; 79 chunks, 3 figures)
  and answered questions. One question was timed: 81 s for the search and 329 s for the
  whole answer. The code ran as it was written: compared with the release, only the
  threshold in `config.yaml` and `machine.md` were changed on that laptop, and both
  changes are part of this release.

**Release 2026-11-23, this one, has not been run with a real model.** It replaces the
chat screen, changes what is sent to the answer model, and restructures ingestion.
Everything in the table marked "stand-ins" was tested with fakes in place of the models.

| Part | Code | How it was checked | Never checked |
|---|---|---|---|
| Configuration, environment check | `config.py`, `check_env.py` | Tests; `check_env` run for real on Laptop B (18 passed, 6 failed: the six are the missing models) | |
| Document identity and manifest | `ingest/documents.py`, `scripts/build_manifest.py` | Tests; the manifest was built from the real RBI index and matched 607 of the 618 PDFs | |
| Docling parsing | `ingest/parser.py` | Tests of the options and the parse cache. Release 2026-11-20 parsed 15 real PDFs on Laptop A | The new choice between parsing with and without OCR, with Docling's real models |
| Text-layer check (`parsing.ocr: auto`) | `ingest/parser.py` | Tests; run over all 618 real PDFs on Laptop B: 7 have a scanned page | |
| Reading Docling's output | `ingest/elements.py`, `ingest/tables.py` | Tests on Docling documents built by hand. Release 2026-11-20 on 15 real PDFs | Which real pictures the new size rules skip |
| Personal-data check | `ingest/screening.py` | Tests | Its false-alarm rate on real circulars |
| Figure descriptions | `ingest/figures.py` | Tests with a faked model server. Release 2026-11-20 described 3 real figures | Descriptions of the scaled-down images |
| Chunking | `ingest/chunker.py` | Tests with words counted as tokens | Sizes with BGE-M3's real tokenizer |
| Ingestion in steps | `ingest/pipeline.py` | Tests with stand-ins | Whether memory is really freed between the steps |
| Embedding and reranking | `index/embedder.py`, `retrieve/reranker.py` | Release 2026-11-20 on Laptop A | The 8-bit option (`quantize.py`): tested on a small network only |
| Vector store | `index/store.py` | Tests against the real Qdrant client; timed with 79, 3,000 and 20,000 passages (`machine.md`) | |
| Retrieval | `retrieve/__init__.py` | Tests: real Qdrant, stand-in models | Search quality |
| Answers and citations | `generate/` | Tests with a faked model server | Whether the answer model writes the new short citations (`[E1, p. 12]`) |
| Chat screen | `web/` | Tests through the server's own addresses, stand-in models. Opened in a browser on Laptop B with stand-in models | With real answers |
| Statistics | `stats.py`, `db.py` | Tests | On a real query log |
| Evaluation | `evaluate.py` | Tests with stand-ins | On real questions. `eval/questions.csv` has no rows yet |
| Comparing an older copy | `scripts/compare_with_release.py` | Tests; run on the 2026-11-20 zip and on the code from Laptop A | |

Test counts on 2026-11-23: 460 fast tests pass on Laptop B (`pytest -m "not slow"`). 15
tests need the real models (`pytest -m slow`, in `tests/test_real_models.py`). None of
the 15 has been run on Laptop B, and it is not recorded whether the first 11 were run on
Laptop A. A failure there may be a mistake in the test as easily as in the code.

## What changed on 2026-11-23

1. **The chat screen is the project's own web page**, not Streamlit:
   `python -m docqa.web --open`. On the left it lists the ingested documents by a short
   name (point at one for the full title, file, version, date, pages and passages) and
   shows the figures: documents, pages, passages, questions asked, typical times. The
   passages found for a question appear before the answer starts. Each answer shows how
   long every step took. Stop ends an answer.
2. **Models are loaded when the screen starts** (`ui.warm_up`), in the background, so
   the first question does not pay for loading them.
3. **The answer model reads less.** Of the six passages kept, only those that score at
   least half as well as the best one are sent (`generation.evidence_score_ratio`). The
   sources list says which ones were not sent.
4. **The answer model writes less.** It cites `[E1, p. 12]`, and the code turns that
   into `[document title, p. 12]`. It no longer copies titles of up to 344 characters.
5. **Every step of every question is timed** and stored in the query log.
   `python -m docqa.stats` prints the typical times.
6. **Ingestion works in steps over groups of files** (`ingestion.group_files`): read
   all, then describe all figures, then embed and store. Only one large model is in
   memory at a time, which matters with 15.7 GB of RAM.
7. **OCR runs only for documents that have a scanned page** (`parsing.ocr: auto`). Of
   the 618 sample PDFs, 7 have one.
8. **Decorative pictures are skipped and figure images are scaled down** before the
   vision model sees them (`figures.min_side_px`, `max_aspect_ratio`, `max_side_px`).
9. **Optional 8-bit models** (`embedding.quantize`, `reranker.quantize`). Off.
10. **The vision model can have its own server address** (`llm.vision_base_url`), and
    the screen and `check_env` say so when a model runs on another machine.
11. **`python -m docqa.ingest --ingested-only`** brings the documents already in the
    index up to date and leaves new files alone.
12. **Streamlit is no longer installed.** `starlette`, `uvicorn`, `pypdfium2`, `pillow`
    and `psutil` are pinned instead. All five were already installed, as dependencies.

After the update every ingested document is indexed again once, because the way a
document's passages are recorded changed. Parse results and figure descriptions should
come from the caches, which makes this the embedding step alone.

## Where the time of an answer goes

Measured on Laptop A with release 2026-11-20, for one question: the search took 81 s
and the whole answer 329 s. So the answer model took about 248 s.

**The database is not the slow part.** Timed on Laptop B on 2026-11-23, with no model
loaded, through the project's own search code:

| Passages in the index | One database search | Share of the 81 s |
|---|---|---|
| 79 (what Laptop A has) | 6 ms | less than a hundredth of one percent |
| 3,000 | 0.2 s | |
| 20,000 | 1.6 s | |

Writing a question to the SQLite log took 14 ms. `machine.md` has the details.

So the 81 s is the two search models on the processor: BGE-M3 reads the question, and
the reranker reads the question against each of 30 passages. For the first question
after a start, loading both models is in there as well. The 248 s is the answer model
on the processor: reading the six passages, then writing. Release 2026-11-20 did not
record how these split. This release does, per question, on the screen and in
`python -m docqa.stats`.

**Why MongoDB or Atlas would not make it faster.** A different database replaces the
6 ms, not the minutes. It would also break two rules of the project. Atlas is a cloud
service, so the text of the documents would be stored outside the machine. MongoDB
itself is a server program to install and run, where Qdrant's local mode and SQLite are
libraries inside Python. If the index grows towards 20,000 passages the search does get
slower, and the step for that is already decided in `CLAUDE.md`: a Qdrant server,
through `qdrant.url`.

## Making it faster

Nothing in this section has been measured with real models. The expected effects are
estimates. Measure on Laptop A, change one thing at a time, and keep what helps.

**Measure first.** Ask the same three questions before and after each change. The
screen shows the time of every step under each answer, and this prints the medians:

```powershell
python -m docqa.stats
```

**Already switched on in this release, at no cost to quality that is known:**

| What | Setting | What it saves |
|---|---|---|
| Models loaded at start | `ui.warm_up: true` | The loading time, off the first question |
| Short citations | none (`generate/prompt.py`) | The model writes a few characters per citation, not a title |
| Stop button | none | A wait that is not worth it |

**Already switched on, with a cost:**

| What | Setting | What it saves | What it costs |
|---|---|---|---|
| Weak passages are not sent | `generation.evidence_score_ratio: 0.5` | Reading time of the answer model, the largest part of an answer | A fact that is only in a weak passage is not available to the model. `0` sends all six |
| Evidence budget | `generation.evidence_char_budget: 12000` | The same | Long evidence is cut, and the answer says so |

**To try, most promising first:**

1. **Keep Ollama's model in memory.** Ollama unloads a model that has not been used for
   five minutes, and the next question waits for it to load again. Set the user
   environment variable `OLLAMA_KEEP_ALIVE` to `30m` and restart Ollama. (From Ollama's
   documentation as remembered; not tried.)

   ```powershell
   [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "User")
   ```

2. **Rerank fewer candidates:** `retrieval.candidates: 15`. Reranking time is in
   proportion to this number, and reranking is most of the search time. Cost: a
   passage that the first search ranked 16th to 30th is no longer found. Check with
   `python -m docqa.evaluate --retrieval-only` before and after, once
   `eval/questions.csv` has rows. `CLAUDE.md` names 30, so this is the project owner's
   decision; `tests/test_config.py` guards the value.
3. **8-bit reranker:** `reranker.quantize: true`. Expect the reranker to run in roughly
   half the time or less. Its scores change, so choose the threshold again
   (`python -m docqa.evaluate --suggest-threshold`). Check first with
   `pytest -m slow -k 8_bit -s`, which prints the times of both variants.
4. **8-bit embedding model:** `embedding.quantize: true`. Shortens reading the question
   and, much more, ingestion. The vectors change, so every document is embedded again
   on the next ingest run, and the stored vectors and the question's must be made with
   the same setting.
5. **Fewer passages for the answer model:** `retrieval.top_k: 4`, or a smaller
   `generation.evidence_char_budget`. Cost: less evidence. `CLAUDE.md` names 6.
6. **Shorter answers:** `llm.max_tokens: 400`. Only matters when answers are long. An
   answer that is cut off says so.
7. **Shorter reranker input:** `reranker.max_length: 512`. The reranker then judges a
   long passage by its first part only. `tests/test_config.py` guards against this on
   purpose, so treat it as a last resort.

For ingestion:

1. `--no-figures` leaves the vision model out. Figures are indexed by their captions.
2. `parsing.table_mode: fast` uses Docling's quicker table model. Tables with merged
   cells may come out worse.
3. `embedding.quantize: true`, as above.
4. `ingestion.group_files` sets how many files are read before the models change. A
   larger group means fewer model loads; a smaller one means results sooner.

**What settings cannot do.** On a laptop processor a 7B model reads a prompt at some
tens of tokens per second and writes a few tokens per second. An answer from three to
six passages therefore takes minutes, whatever the settings. Answers in seconds need
other hardware: a GPU server, which is the plan in `CLAUDE.md`, or a hosted endpoint.

## Answers from a hosted endpoint

**This is off, and switching it on is the project owner's decision.** `CLAUDE.md` says
"Nothing leaves the machine. No cloud AI APIs", and that rule stands until its owner
changes it. Check also what the rules of your organisation allow for the documents
concerned. Use it with public documents only.

What it is: the answer model runs at a provider, behind the same kind of address that
Ollama offers (an "OpenAI-compatible endpoint"). The code needs no change.

What is sent to the provider with every question: the question, the passages chosen as
evidence (text from the documents), and the instructions. What stays on the machine:
the documents, the index, the search models, the query log, and the figures, provided
`vision_base_url` points at this machine as below.

```yaml
llm:
  base_url: https://<the provider's address>/v1
  allow_remote: true
  api_key_env: DOCQA_LLM_API_KEY
  vision_base_url: http://localhost:11434/v1   # figures stay with Ollama on this machine
  answer_model: <the provider's name for Qwen2.5 7B Instruct>
```

The key goes in an environment variable, never in a file:

```powershell
[Environment]::SetEnvironmentVariable("DOCQA_LLM_API_KEY", "<the key>", "User")
```

Then `python -m docqa.check_env` checks both servers and prints a note that says what
is sent where. The chat screen shows "Answers <model> on <host>".

Providers that a web search on 2026-11-23 listed for Qwen2.5 7B Instruct. None was
tried, and model lists and prices change:

| Provider | Address | Model name there |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen-2.5-7b-instruct` |
| Novita AI | `https://api.novita.ai/v3/openai` | `qwen/qwen2.5-7B-Instruct` |
| Alibaba Cloud Model Studio, Singapore | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen2.5-7b-instruct` |
| DeepInfra | see its documentation | see its model list |

Before choosing one:

- Stay with the same model. `CLAUDE.md` lists the models that may be used, and the 3B
  and 72B sizes of Qwen2.5 have licences that do not allow commercial use.
- Read the provider's terms: whether prompts are stored, and whether they are used for
  training. Choose one that does neither.
- The threshold does not change: the reranker still runs on the machine.

What it would change, as an estimate: the answer model's 248 s would become seconds.
The 81 s of search would stay, so the settings of the previous section still matter.

A GPU server of your own with vLLM is the other route, and the one `CLAUDE.md` plans.
It uses the same four settings, and nothing goes to a third party. vLLM serves one
model per server, which is what `vision_base_url` is for.

## Updating the copy on Laptop A

`COPILOT_UPDATE.md` is this section as a task for GitHub Copilot.

Stop the old chat screen first. Then get the code, in one of three ways:

**a. From the git bundle** (a single file, `docqa-update-2026-11-23.bundle`). It holds
this release as a branch on top of the commit that Laptop A pushed to GitHub. In the
project folder on Laptop A:

```powershell
git status
```

Commit anything it lists, then:

```powershell
git fetch "<path to>\docqa-update-2026-11-23.bundle" speed-and-new-ui:speed-and-new-ui
```

```powershell
git checkout speed-and-new-ui
```

**b. From GitHub**, if the branch has been pushed there: `git fetch origin`, then
`git checkout speed-and-new-ui`.

**c. From the zip** (`docqa-code-2026-11-23.zip`), if the project folder is not a git
repository. Do not extract it over the old folder. Extract it to a new folder, run
`python scripts\compare_with_release.py "<old folder>"` there to see what the old copy
changed, move `data\` and `models\` from the old folder to the new one, and activate
the old virtual environment.

Then, whichever way was used:

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install --no-deps -e .
```

The first command installs the pinned versions of `starlette` and `uvicorn` if the
environment has others. It downloads a few small packages at most.

```powershell
python -m docqa.check_env
```

```powershell
pytest -m "not slow"
```

460 should pass. Bring the ingested documents up to date, without starting on the PDFs
that were never ingested:

```powershell
python -m docqa.ingest "D:\EY Internship\docqa-docs-rbi-public-2026-11-20\docqa\docs" --ingested-only
```

It should say that 15 documents are to be ingested and about 600 were left alone, and
each figure line should say "cached". Then start the screen and ask the question that
took 329 s:

```powershell
python -m docqa.web --open
```

Note the times under the answer, then follow "Making it faster". The real-model tests
that are new in this release:

```powershell
pytest -m slow -k "without_ocr or 8_bit or chat_screen" -s
```

## Setting up on a new machine

Run everything from the project folder, in PowerShell.

### 1. Copy the project

Copy the whole folder except `data/`, which is rebuilt. The PDFs in `docs/` are not in
git, so a `git clone` alone does not bring them: copy `docs/` as well. Take only public
or sanitised documents.

### 2. Python environment

Python 3.11 or newer. The pins in `requirements.txt` were resolved for Python 3.14 on
Windows, and installed as they are on Python 3.13.3 on Laptop A. On another version pip
may not find a wheel for an exact pin: if so, stop and decide with the project owner.
Do not loosen pins silently.

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install --no-deps -e .
```

`.venv/` is git-ignored. If the project folder is synced by OneDrive or similar, put
the environment outside it.

### 3. Ollama

It needs both models, a larger context window, and its cloud features switched off.

```powershell
ollama list
```

Both `qwen2.5:7b` and `qwen2.5vl:7b` must be listed. Then set two user environment
variables and restart Ollama (quit it from its icon near the clock, start it again):

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "8192", "User")
```

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_NO_CLOUD", "1", "User")
```

Why the context window matters: Ollama's default is 4,096 tokens, and it drops whatever
does not fit without saying so. Six chunks of up to 800 tokens plus the instructions
need more. The slow test `test_ollama_context_window_is_large_enough_for_six_chunks`
checks the result. `machine.md` records both variables as set on Laptop A.

### 4. Download the other models, once

These commands use the network, on purpose. The pipeline itself never downloads
anything: importing `docqa` puts the Hugging Face libraries in offline mode.

```powershell
hf download BAAI/bge-m3 --exclude "onnx/*"
```

```powershell
hf download BAAI/bge-reranker-v2-m3
```

```powershell
docling-tools models download layout tableformer rapidocr -o models/docling
```

The first two go to the Hugging Face cache, where `config.yaml` expects them. The third
fills `models/docling`, which `parsing.artifacts_path` points at. About 5 GB in all
(an estimate).

Before relying on them, open each model's page and record its licence in `machine.md`.
`CLAUDE.md` requires licences that allow commercial use. From memory, and not checked:
BGE-M3 is MIT and the reranker is Apache 2.0. Docling's layout, table and OCR models
have not been checked at all.

### 5. Check

```powershell
python -m docqa.check_env
```

Every line should say PASS.

```powershell
pytest -m "not slow"
```

460 should pass. Then the real-model tests, one group at a time, in this order:

```powershell
pytest -m slow -k "ollama or answer_model or vision"
```

```powershell
pytest -m slow -k embedding
```

```powershell
pytest -m slow -k reranker
```

```powershell
pytest -m slow -k docling
```

```powershell
pytest -m slow -k cited_answer
```

```powershell
pytest -m slow -k "without_ocr or 8_bit or chat_screen" -s
```

### 6. First documents

Start with two or three. Parsing is slow on a CPU.

```powershell
python -m docqa.ingest docs --limit 3
```

Look at what came out before going further: `data/parsed/` (Docling's output),
`data/figures/` (the crops), and the warnings the command prints. The last line says
how long each of the three steps took.

```powershell
python -m docqa.ingest --list
```

### 7. Questions and the threshold

Write 10 to 20 questions about the ingested documents in `eval/questions.csv`
(`eval/README.md` explains the columns). Include several that no document answers, with
type `not_in_docs`: the threshold is chosen from them.

```powershell
python -m docqa.evaluate --suggest-threshold
```

Put the proposed value in `config.yaml` as `retrieval.rerank_score_threshold`, in place
of the temporary 0.50, then:

```powershell
python -m docqa.evaluate
```

Open `results.csv` in the new folder under `eval/results/` and fill in `manual_correct`.

### 8. The chat screen

```powershell
python -m docqa.web --open
```

It listens on this machine only, at http://127.0.0.1:8000. Stop it (Ctrl+C) before
ingesting or evaluating again. All three use the same index, and Qdrant's local mode
lets only one program open it.

## Where it is most likely to need fixing

In rough order of likelihood. Each item names the file to look at. Items 1 to 5 are
new in release 2026-11-23, which has not met a real model.

1. **The short citations** (`generate/prompt.py`, `generate/citations.py`). The model is
   told to cite `[E1, p. 12]`. A 7B model may write titles instead, or leave the page
   out. A title that fits one retrieved document, and `[E1]` alone, are both accepted.
   If many answers end with "could not be checked", the instruction needs work. Raise
   `PROMPT_VERSION` when you change it.
2. **The chat screen with real answers** (`web/server.py`, `web/static/app.js`). An
   answer travels as a stream of events over one long request. Watch for an answer that
   stops halfway, for Stop not stopping the model, and for the screen saying "Loading
   the search models" for ever (the server's log says why).
3. **Parsing without OCR** (`ingest/parser.py`). A document counts as needing OCR when
   a page holds an image and at most `parsing.scanned_page_max_chars` characters of
   text. A scan with a poor hidden text layer is then parsed without OCR and comes out
   as garbage: set `parsing.ocr: true` for such a set.
4. **Memory between the ingestion steps** (`ingest/pipeline.py`,
   `parser.release_converters`). Docling's models are dropped before the vision model is
   called, and the embedding model between groups. Whether the memory really returns
   to the system was not observed. Watch Task Manager during a run.
5. **The 8-bit option** (`quantize.py`). It uses a PyTorch function that is marked as
   deprecated. If it fails, the model runs as before and the log says so.
6. **Docling with real models** (`ingest/parser.py`). OCR in particular: the only
   engine installed is RapidOCR with its PyTorch backend. It is not recorded whether a
   scanned page was among the 15 PDFs parsed so far.
7. **What Docling returns for these PDFs** (`ingest/elements.py`). The extraction was
   written against hand-built documents. Check on a real one: heading levels (Docling
   may give every heading level 1, which makes `heading_path` shallow), list markers,
   captions, page headers and footers, and tables that continue on the next page.
8. **The vision model's reply** (`ingest/figures.py`). The instruction asks for one
   JSON object. Unreadable replies are kept as raw text and still indexed; if that
   happens often, the instruction needs work. Raise `PROMPT_VERSION` when you change it,
   or cached descriptions will be reused.
9. **Chunk sizes with the real tokenizer** (`ingest/chunker.py`). 79 chunks from 15
   PDFs is about five per document, which is few. Look at the token counts once.
10. **Near-duplicate documents.** The Master Directions exist once per type of bank. The
    top six passages for a question may be six near-identical ones. The document title
    is part of what is embedded, which should help; it has not been measured.

## Decisions taken while building

`design.md` section 16 listed these as open. They were settled as follows so that the
code could be written. Each is easy to change.

| Question | What the code does | Where |
|---|---|---|
| Where titles, versions and dates come from | `manifest.csv` in the project root, one row per file. Built from the RBI index: 607 documents have their full title and date; the 11 others use their file name | `paths.manifest`, `ingest/documents.py` |
| What identifies a document | Its path below `docs/`. An edited file replaces its chunks; a renamed file is a new document | `ingest/documents.py` |
| Which page is cited | Chunks record where each later page begins, and the model is shown those places, so it can cite the page a fact is on | `Chunk.page_marks`, `generate/prompt.py` |
| How the model names a source | By a label, `E1` to `E6`, with the page. The code replaces the label with the document's title | `generate/citations.py` |
| Which passages the model gets | Of the six kept, those scoring at least half as well as the best, within the evidence budget | `generation.evidence_score_ratio`, `generation.evidence_char_budget` |
| What is embedded | The document title, then the headings above the chunk, then its text | `index.searchable_text` |
| Scale of rerank scores | Normalised, 0 to 1. The threshold is on that scale | `reranker.normalize` |
| When OCR runs | Only for a document that has a page with an image and almost no text | `parsing.ocr`, `ingest/parser.py` |
| Decorative pictures | Skipped without a caption if they cover under 2% of the page, are under 60 pixels on a side, or are 8 times longer than wide. A figure the vision model calls a logo, photograph, signature or stamp is not indexed unless something readable is in it | `figures`, `ingest/elements.py`, `ingest/figures.py` |
| A figure that was not described | Indexed by its caption alone, if it has one | `ingest/chunker.py` |
| Real customer data | A file with 3 or more PAN-like or Aadhaar-like numbers is not ingested | `screening`, `ingest/screening.py` |
| A table too long for the models | Never split. Search uses its first 1,024 tokens; the answer model gets as much as fits the evidence budget, marked as cut | `embedding.max_length`, `generation.evidence_char_budget` |
| Table of contents | Not indexed | `ingest/elements.py` |
| More than one question at a time | The second waits for the first. The models use the whole processor anyway | `web/server.py` |
| The chat screen | A page of plain HTML, CSS and JavaScript served by Starlette and uvicorn on this machine. It loads nothing from the internet | `web/` |

Still open:

- A table that continues on the next page arrives as two tables and stays two.
- Follow-up questions. Each question is answered on its own.
- Where `data/` lives when the project folder is synced by OneDrive.
- Which 10 to 20 documents form the prototype set.
- Whether answers may come from a hosted endpoint (see that section).

## Using an AI coding assistant on this project

`AGENTS.md` holds the rules for one. `COPILOT_PROMPT.md` is a task for GitHub Copilot
in VS Code that sets the build up on a machine with the models, and `COPILOT_UPDATE.md`
is the task that brings an existing copy up to this release.

One thing to keep in mind. The pipeline sends nothing off the machine, but a cloud
assistant does: whatever it reads, including file contents and command output, goes to
its provider. That is fine for this code and for public documents. Do not let it open
documents that are not public, anything in `data/` derived from them, or a query log
that holds real questions.

The same care applies to a repository on GitHub. The project's repository there is
public. It holds code and notes only. Keep it that way: no documents, nothing from
`data/`, no keys, and no notes about a company machine (`machine.local.md` is kept out
of git for that reason).

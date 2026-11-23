# Handover: finishing the build on a machine that has the models

Written on 2026-11-20, on a laptop that has none of the models.

Read this before running anything. It says what exists, what has and has not been
checked, how to set the project up on another machine, and where it is most likely to
need fixing. `CLAUDE.md` holds the rules of the project, `design.md` explains the
methods, and `README.md` lists the commands.

## Where things stand

All of the pipeline code is written: ingestion, indexing, retrieval, cited answers, the
chat screen, the query log and the evaluation script.

**None of it has been run with a real model.** The laptop it was written on has no
Ollama program, no BGE-M3, no reranker and none of Docling's models. Everything below
marked "stand-ins" was tested with small fakes in place of the models.

| Part | Code | How it was checked | Never checked |
|---|---|---|---|
| Configuration, environment check | `config.py`, `check_env.py` | Tests; `check_env` run for real (14 passed, 6 failed: the six are the missing models) | |
| Document identity and manifest | `ingest/documents.py`, `scripts/build_manifest.py` | Tests; the manifest was built from the real RBI index and matched 607 of the 618 PDFs | |
| Docling parsing | `ingest/parser.py` | Tests of the options and the parse cache only | Converting a real PDF. Docling's models were not available |
| Reading Docling's output | `ingest/elements.py`, `ingest/tables.py` | Tests on Docling documents built by hand with the real `docling-core` classes | What Docling really returns for these PDFs |
| Personal-data check | `ingest/screening.py` | Tests | Its false-alarm rate on real circulars |
| Figure descriptions | `ingest/figures.py` | Tests with a faked model server | The vision model itself, and whether it returns usable JSON |
| Chunking | `ingest/chunker.py` | Tests with words counted as tokens | Sizes with BGE-M3's real tokenizer |
| Embedding and reranking | `index/embedder.py`, `retrieve/reranker.py` | Call signatures read from the installed library | Loading or running either model |
| Vector store | `index/store.py` | Tests against the real Qdrant client in local mode | Speed with thousands of chunks |
| Retrieval | `retrieve/__init__.py` | Tests: real Qdrant, stand-in models | Search quality |
| Answers and citations | `generate/` | Tests with a faked model server | What the answer model really writes |
| Chat screen | `app/main.py` | Tests with Streamlit's own test harness, stand-in models | In a browser with real answers |
| Evaluation | `evaluate.py` | Tests with stand-ins | On real questions. `eval/questions.csv` has no rows yet |

Test counts on 2026-11-20: 347 fast tests pass (`pytest -m "not slow"`). 11 tests need
the real models (`pytest -m slow`, in `tests/test_real_models.py`). They have never
been run, so a failure there may be a mistake in the test as easily as in the code.

## Setting up on the new machine

Run everything from the project folder, in PowerShell.

### 1. Copy the project

Copy the whole folder except `data/`, which is rebuilt. The PDFs in `docs/` are not in
git, so a `git clone` alone does not bring them: copy `docs/` as well. Take only public
or sanitised documents.

### 2. Python environment

Python 3.11 or newer. The pins in `requirements.txt` were resolved for Python 3.14 on
Windows. On another version pip may not find a wheel for an exact pin: if so, stop and
decide with the project owner. Do not loosen pins silently.

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
the environment outside it (see `machine.md` for why).

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
need more. The setting names come from Ollama's documentation and source, read on
2026-11-18 and 2026-11-19; setting them has not been tried. The slow test
`test_ollama_context_window_is_large_enough_for_six_chunks` checks the result.

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

Every line should say PASS. A note about the unset rerank threshold is expected.

```powershell
pytest -m "not slow"
```

347 should pass. Then the real-model tests, one group at a time, in this order:

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

### 6. First documents

Start with two or three. Parsing is slow on a CPU.

```powershell
python -m docqa.ingest docs --limit 3
```

Look at what came out before going further: `data/parsed/` (Docling's output),
`data/figures/` (the crops), and the warnings the command prints.

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

Put the proposed value in `config.yaml` as `retrieval.rerank_score_threshold`, then:

```powershell
python -m docqa.evaluate
```

Open `results.csv` in the new folder under `eval/results/` and fill in `manual_correct`.

### 8. The chat screen

```powershell
streamlit run src/docqa/app/main.py
```

Stop it before ingesting or evaluating again. All three use the same index, and Qdrant's
local mode lets only one program open it.

## Where it is most likely to need fixing

In rough order of likelihood. Each item names the file to look at.

1. **Docling with real models** (`ingest/parser.py`). Whether `models/docling` has the
   layout Docling expects for `artifacts_path`. Whether OCR runs: the only OCR engine
   installed is RapidOCR with its PyTorch backend, and its model files were never
   tested. If OCR is the only thing failing, set `parsing.ocr: false` to get digital
   PDFs through, and treat scanned pages as a separate job.
2. **What Docling returns for these PDFs** (`ingest/elements.py`). The extraction was
   written against hand-built documents. Check on a real one: heading levels (Docling
   may give every heading level 1, which makes `heading_path` shallow), list markers,
   captions, page headers and footers, and tables that continue on the next page.
3. **FlagEmbedding calls** (`index/embedder.py`, `retrieve/reranker.py`). The arguments
   were read from the installed library, not tried. Watch for an argument it rejects,
   for the reranker returning a single number for a single pair, and for load time.
4. **The vision model's reply** (`ingest/figures.py`). The instruction asks for one
   JSON object. A 7B model may not comply. Unreadable replies are kept as raw text and
   still indexed; if that happens often, the instruction needs work. Raise
   `PROMPT_VERSION` when you change it, or cached descriptions will be reused.
5. **Citations from the answer model** (`generate/prompt.py`, `generate/citations.py`).
   The RBI titles are long, up to 344 characters, and the model has to copy them. A
   shortened title is accepted when it fits exactly one retrieved document, and so is a
   citation by evidence number. If many answers still end with "could not be checked",
   give the documents shorter titles in `manifest.csv`, or change the instruction.
   Raise `PROMPT_VERSION` in `generate/prompt.py` when you change the wording.
6. **Chunk sizes with the real tokenizer** (`ingest/chunker.py`). The rules were tested
   with words as tokens. Look at the token counts of real chunks once.
7. **Near-duplicate documents.** The Master Directions exist once per type of bank. The
   top six passages for a question may be six near-identical ones. The document title
   is part of what is embedded, which should help; it has not been measured.
8. **Speed.** Nothing has been timed. The ingest command prints seconds per step. If
   reranking 30 candidates is too slow, `retrieval.candidates` is the setting to lower.

## Decisions taken while building

`design.md` section 16 listed these as open. They were settled as follows so that the
code could be written. Each is easy to change.

| Question | What the code does | Where |
|---|---|---|
| Where titles, versions and dates come from | `manifest.csv` in the project root, one row per file. Built from the RBI index: 607 documents have their full title and date; the 11 others use their file name | `paths.manifest`, `ingest/documents.py` |
| What identifies a document | Its path below `docs/`. An edited file replaces its chunks; a renamed file is a new document | `ingest/documents.py` |
| Which page is cited | Chunks record where each later page begins, and the model is shown those places, so it can cite the page a fact is on | `Chunk.page_marks`, `generate/prompt.py` |
| What is embedded | The document title, then the headings above the chunk, then its text | `index.searchable_text` |
| Scale of rerank scores | Normalised, 0 to 1. The threshold is on that scale | `reranker.normalize` |
| Decorative pictures | A picture smaller than 2% of its page with no caption is skipped. A figure the vision model calls a logo, photograph, signature or stamp is not indexed unless something readable is in it | `figures.min_page_fraction`, `ingest/figures.py` |
| A figure that was not described | Indexed by its caption alone, if it has one | `ingest/chunker.py` |
| Real customer data | A file with 3 or more PAN-like or Aadhaar-like numbers is not ingested | `screening`, `ingest/screening.py` |
| A table too long for the models | Never split. Search uses its first 1,024 tokens; the answer model gets as much as fits the evidence budget, marked as cut | `embedding.max_length`, `generation.evidence_char_budget` |
| Table of contents | Not indexed | `ingest/elements.py` |

Still open:

- A table that continues on the next page arrives as two tables and stays two.
- Follow-up questions. Each question is answered on its own.
- Where `data/` lives when the project folder is synced by OneDrive.
- Which 10 to 20 documents form the prototype set.

## Using an AI coding assistant on this project

`AGENTS.md` holds the rules for one, and `COPILOT_PROMPT.md` is a ready-made task for
GitHub Copilot in VS Code.

One thing to keep in mind. The pipeline sends nothing off the machine, but a cloud
assistant does: whatever it reads, including file contents and command output, goes to
its provider. That is fine for this code and for public documents. Do not let it open
documents that are not public, anything in `data/` derived from them, or a query log
that holds real questions.

# BFSI document Q&A prototype

A chatbot that runs on one machine and answers questions from banking documents:
policies, circulars, annual reports and process manuals. Every answer cites the
document and the page it came from. It reads tables, charts, flowcharts and scanned
pages as well as plain text, and it says "I could not find this in the documents."
when the documents do not hold the answer.

Nothing leaves the machine. The models run locally, through Ollama and in Python.

## Status

The code is complete and its fast tests pass. **It has not yet been run with the real
models.** `HANDOVER.md` says what was checked, what was not, and how to set it up on a
machine that has the models.

## How it works

Ingestion, once per document:

1. The file is hashed. An unchanged file is skipped.
2. Docling parses the PDF into headings, paragraphs, lists, tables and pictures, each
   with its page. Scanned pages go through OCR.
3. Text is cut into chunks along the document's headings. A table is one chunk. A
   figure is described once by the vision model, and the description is one chunk.
4. BGE-M3 turns each chunk into a dense and a sparse vector.
5. Chunks and vectors go into Qdrant. A register of documents goes into SQLite.

A question:

1. The question is embedded. A dense and a sparse search each return candidates.
2. The two lists are merged by reciprocal rank fusion. The best 30 are reranked, and
   the best 6 are kept.
3. If even the best is weak, the chatbot says it found nothing. The answer model is
   not called.
4. Otherwise the answer model writes the answer from the 6 chunks and cites each
   statement as `[document title, p. N]`.
5. Every citation is checked against the chunks the model was given. The question,
   the chunks, the answer and the time taken are logged.

`design.md` explains each step and why it is done that way.

## Commands

Run from the project folder with the virtual environment active.

| Command | What it does |
|---|---|
| `python -m docqa.check_env` | Says PASS or FAIL for Python, the packages, Ollama, every model and Qdrant |
| `pytest -m "not slow"` | Fast tests. They need no model |
| `pytest -m slow` | Tests that need the real models |
| `python -m docqa.ingest docs --limit 3` | Ingests up to three documents that are new or changed |
| `python -m docqa.ingest --list` | Lists the ingested documents |
| `python -m docqa.evaluate --suggest-threshold` | Proposes the refusal threshold from the test questions |
| `python -m docqa.evaluate` | Runs the test questions and writes the results |
| `streamlit run src/docqa/app/main.py` | Starts the chat screen |
| `python scripts/build_manifest.py --force` | Rebuilds `manifest.csv` from the RBI index workbook |

Options of the ingest command:

| Option | Meaning |
|---|---|
| `--limit N` | Process at most N files in this run |
| `--no-figures` | Do not call the vision model. Figures are cropped but not described |
| `--force` | Ingest again even if nothing changed |
| `--yes` | Do not stop to ask when more than 25 files need work |
| `--prune` | Also remove documents whose files are gone |
| `--allow-flagged` | Ingest a file that the personal-data check flagged. Look at the file first |

Only one program can have the index open at a time. Stop the chat screen before
ingesting or evaluating.

## Folders

```
docs/                 sample documents (input; the pipeline never changes them)
manifest.csv          title, version, effective date and access group per document
eval/questions.csv    test questions
eval/results/         one folder per evaluation run
data/                 generated: parse cache, figure crops, Qdrant index, SQLite file
models/               Docling's models (downloaded once; not in git)
src/docqa/            the code: ingest/ index/ retrieve/ generate/ app/ and evaluate.py
tests/                fast tests, and tests/test_real_models.py for the real models
scripts/              one-off helpers
config.yaml           every model name, path, size and threshold
```

`data/`, `models/` and the PDFs are kept out of git.

## Documents to read

| File | What it is for |
|---|---|
| `HANDOVER.md` | Current state, setup on a new machine, and where problems are most likely |
| `CLAUDE.md` | The decisions and rules of the project |
| `design.md` | The methods, explained in detail |
| `machine.md` | The machine the code was written on, and traps found in the installed software |
| `AGENTS.md` | Rules for AI coding assistants |
| `COPILOT_PROMPT.md` | A ready-made task for GitHub Copilot: get the build running with the real models |
| `eval/README.md` | How to write test questions |
| `docs/README.md` | What may go into `docs/` |

## Configuration

Everything adjustable is in `config.yaml`, with a comment on each setting. A key that
is misspelt or missing stops the program with a message that names it.

One setting is deliberately empty: `retrieval.rerank_score_threshold`. It decides when
the chatbot says it found nothing, and a sensible value depends on the documents. The
chatbot does not answer until it is set. `python -m docqa.evaluate --suggest-threshold`
proposes a value.

## The manifest

`manifest.csv` gives each document the title that appears in citations, and the
effective date that is shown when two documents disagree:

```
file,doc_title,version,effective_date,access_group
Master Circulars/Example.pdf,Master Circular - Example,2026-27/12,2026-04-01,
```

`file` is the path below `docs/`. A document that is not listed gets its file name as
its title and no date. Changing a row re-indexes that document on the next ingest run.

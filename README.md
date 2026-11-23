# BFSI document Q&A prototype

A chatbot that runs on one machine and answers questions from banking documents:
policies, circulars, annual reports and process manuals. Every answer cites the
document and the page it came from. It reads tables, charts, flowcharts and scanned
pages as well as plain text, and it says "I could not find this in the documents."
when the documents do not hold the answer.

Nothing leaves the machine. The models run locally, through Ollama and in Python. (An
answer model on another server can be configured. It is off, and switching it on is
the project owner's decision: see `HANDOVER.md`, "Answers from a hosted endpoint".)

## Status

The code is complete and its fast tests pass.

- Release 2026-11-20 was set up on a second laptop that has the models. It ingested 15
  PDFs and answered questions there, slowly: one answer took between five and seven
  minutes.
- Release 2026-11-23 (this one) is about that: a new chat screen, shorter prompts,
  faster ingestion, and a record of where the time of each answer goes. **It has not
  yet been run with the real models.**

`HANDOVER.md` says what was checked and what was not, how to set the project up, how to
update an existing copy, and what to try to make it faster.

## How it works

Ingestion, once per document:

1. The file is hashed. An unchanged file is skipped.
2. Docling parses the PDF into headings, paragraphs, lists, tables and pictures, each
   with its page. A document with a scanned page goes through OCR; the others do not.
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
4. Otherwise the answer model gets the passages among the 6 that scored well enough,
   writes the answer from them and cites each statement. The screen shows each
   citation as `[document title, p. N]`.
5. Every citation is checked against the passages the model was given. The question,
   the passages, the answer and the time each step took are logged.

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
| `python -m docqa.evaluate --retrieval-only` | Judges the search alone, without the answer model |
| `python -m docqa.evaluate` | Runs the test questions and writes the results |
| `python -m docqa.web --open` | Starts the chat screen at http://127.0.0.1:8000 and opens it in the browser |
| `python -m docqa.stats` | Prints what is indexed and how long questions have been taking |
| `python scripts/build_manifest.py --force` | Rebuilds `manifest.csv` from the RBI index workbook |
| `python scripts/compare_with_release.py <folder>` | Lists what was changed in an older copy of the project, before it is replaced by this one |

Options of the ingest command:

| Option | Meaning |
|---|---|
| `--limit N` | Process at most N files in this run |
| `--no-figures` | Do not call the vision model. Figures are cropped but not described |
| `--force` | Ingest again even if nothing changed |
| `--yes` | Do not stop to ask when more than 25 files need work |
| `--prune` | Also remove documents whose files are gone |
| `--ingested-only` | Bring the documents that were ingested before up to date, and leave new files alone. For use after an update of the code or of a setting |
| `--allow-flagged` | Ingest a file that the personal-data check flagged. Look at the file first |

Options of the chat screen: `--port N`, `--no-warm-up` (do not load the models until
the first question), `--config PATH`.

Only one program can have the index open at a time. Stop the chat screen before
ingesting or evaluating.

## The chat screen

The left side lists the ingested documents by a short name. Pointing at one shows its
full title, file, version, date, pages and passages. Above the list are the figures:
documents, pages, passages, questions asked, and the typical time of an answer and of
each of its parts.

An answer appears word by word. The passages it was written from are listed before the
first word arrives, each with its page, its score and whether it was given to the model.
Tables are shown as tables and figures as pictures. Under each answer is the time every
step took. Stop ends an answer that is taking too long.

## Folders

```
docs/                 sample documents (input; the pipeline never changes them)
manifest.csv          title, version, effective date and access group per document
eval/questions.csv    test questions
eval/results/         one folder per evaluation run
data/                 generated: parse cache, figure crops, Qdrant index, SQLite file
models/               Docling's models (downloaded once; not in git)
src/docqa/            the code: ingest/ index/ retrieve/ generate/ web/, evaluate.py, stats.py
tests/                fast tests, and tests/test_real_models.py for the real models
scripts/              helpers; scripts/releases/ lists the files of each release
config.yaml           every model name, path, size and threshold
```

`data/`, `models/` and the PDFs are kept out of git.

## Documents to read

| File | What it is for |
|---|---|
| `HANDOVER.md` | Current state, setup, updating an existing copy, making it faster |
| `CLAUDE.md` | The decisions and rules of the project |
| `design.md` | The methods, explained in detail |
| `machine.md` | The machine the code was written on, and traps found in the installed software |
| `AGENTS.md` | Rules for AI coding assistants |
| `COPILOT_PROMPT.md` | A task for GitHub Copilot: set the build up on a machine that has the models |
| `COPILOT_UPDATE.md` | A task for GitHub Copilot: bring an existing copy up to this release and measure it |
| `eval/README.md` | How to write test questions |
| `docs/README.md` | What may go into `docs/` |

## Configuration

Everything adjustable is in `config.yaml`, with a comment on each setting. A key that
is misspelt or missing stops the program with a message that names it. Settings marked
SPEED trade time against something else; `HANDOVER.md` says what each one costs.

One setting needs choosing for each set of documents:
`retrieval.rerank_score_threshold`. It decides when the chatbot says it found nothing.
It stands at 0.50, a temporary value that was set for the first 15 documents on the
laptop that runs the models. `python -m docqa.evaluate --suggest-threshold` proposes a
value from the test questions. Left empty (`null`), the chatbot does not answer at all.

## The manifest

`manifest.csv` gives each document the title that appears in citations, and the
effective date that is shown when two documents disagree:

```
file,doc_title,version,effective_date,access_group
Master Circulars/Example.pdf,Master Circular - Example,2026-27/12,2026-04-01,
```

`file` is the path below `docs/`. A document that is not listed gets its file name as
its title and no date. Changing a row re-indexes that document on the next ingest run.

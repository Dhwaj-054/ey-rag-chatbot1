# Task for GitHub Copilot: bring this copy up to release 2026-11-23 and measure it

How to use this file: open the project folder in VS Code, open Copilot Chat in agent
mode, and say: "Read COPILOT_UPDATE.md and carry it out, one step at a time."

## The situation

This machine has the models (Ollama with `qwen2.5:7b` and `qwen2.5vl:7b`, BGE-M3, the
reranker, Docling's models) and a working copy of release 2026-11-20. It ingested 15
PDFs and answers questions, slowly: one answer took 329 s, of which the search took 81 s.

Release 2026-11-23 is about that. It was written on a laptop that has none of the
models, so **none of its changes has run with a real model**. Its 460 fast tests pass
there. `HANDOVER.md`, section "What changed on 2026-11-23", lists the changes. In short:

- a new chat screen (`python -m docqa.web`) in place of Streamlit;
- the answer model reads fewer passages and writes shorter citations;
- every step of every question is timed and stored;
- ingestion works in steps, runs OCR only where a page is scanned, and skips
  decorative pictures;
- optional 8-bit versions of the two search models, switched off.

Your job: get the release onto this machine, show that it works with the real models,
measure where the time goes, and try the speed settings one at a time.

One finding to keep in mind before you start. The database is not what is slow. A
search over the 79 indexed passages takes about 6 ms. The minutes are the embedding
model, the reranker and the answer model, all running on the processor. Do not replace
Qdrant or SQLite.

## Read first

1. `AGENTS.md`: the rules you work under.
2. `HANDOVER.md`, sections "Where the time of an answer goes", "Making it faster" and
   "Updating the copy on Laptop A".
3. `CLAUDE.md`: the project's decisions. Where it says "ask me", ask the user.

## Rules that matter most

- **Nothing leaves the machine.** Do not set `llm.allow_remote` to true, and do not add
  a call to any hosted model or service, unless the user tells you to in this chat.
  `HANDOVER.md` describes the hosted-endpoint option; it is the user's decision.
- **The stack is fixed.** Do not swap, add or upgrade a component or a pinned version
  without asking.
- **One change at a time, measured.** For each speed setting: note the times before,
  change it, ask the same questions, note the times after. Keep it only if it helps
  and the answers are still right.
- **Fix the cause.** Never weaken, skip or delete a test to make it pass.
- **Report plainly.** Say what you ran and what you saw. Say what you did not test.
- **The GitHub repository is public.** Commit code and notes only: no PDFs, nothing
  from `data/`, no keys, no query log.

## Stop and ask the user

- before changing `retrieval.candidates`, `retrieval.top_k` or any other number that
  `CLAUDE.md` names (the user decides; you may measure a trial value first and report);
- before writing a new rerank threshold into `config.yaml`;
- before a run that is likely to take more than about 15 minutes;
- before ingesting documents that were not ingested before;
- before pushing to GitHub.

## Steps

After each step, tell the user in two or three lines what you ran and what you saw.

### 1. Record where this machine stands

Run `python -m docqa.stats` if this copy already has it; otherwise note from the query
log how long the last questions took. Then bring the section "Laptop A" of `machine.md`
up to date by running commands, not from memory: `ollama list`, the Python version,
which models are in the Hugging Face cache and in `models/docling`, free disk space.
That section contradicts itself in places (it says both that Ollama is installed and
that it is not). Do not touch the section "Laptop B".

### 2. Get the release

Stop the old chat screen. Check `git status` and commit anything it lists. Then follow
`HANDOVER.md`, "Updating the copy on Laptop A": from the bundle file, from GitHub, or
from the zip, whichever the user gave you. Afterwards `git log --oneline -3` should show
the release commit on top of this machine's own commit.

### 3. Environment and fast tests

```
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m docqa.check_env
pytest -m "not slow"
```

Every `check_env` line should say PASS, and 460 tests should pass. A failure in the
fast tests on this machine is most likely a difference between Python 3.13 and 3.14, or
a package version: find which.

### 4. Bring the index up to date

```
python -m docqa.ingest "D:\EY Internship\docqa-docs-rbi-public-2026-11-20\docqa\docs" --ingested-only
```

It should find the 15 ingested documents, leave the others alone, take the parse
results from the cache, and say "cached" for each figure. Note the three times in its
last line. If it starts parsing again, stop and find out why the cache was not used
(`src/docqa/ingest/parser.py`, `_cache_is_usable`) before going on.

### 5. The real-model tests that are new

```
pytest -m slow -k "without_ocr or 8_bit or chat_screen" -s
```

Run them one at a time if one fails. They were never run: a failure may be a mistake in
the test as easily as in the code. The 8-bit tests print how long the full and the
8-bit model took; note those four numbers.

### 6. The new screen, and a baseline

```
python -m docqa.web --open
```

Wait for "Ready". Ask the question that took 329 s, and two more: one that a table
answers, and one that the documents cannot answer. For each, note the times shown under
the answer (question read, database, ranking, first word, writing, total) and check:

- the passages appear before the answer starts, and the answer arrives word by word;
- each citation is a document title with a page, and clicking it opens the passage;
- a table is shown as a table, and a figure as a picture with its warning;
- the unanswerable question is refused without the answer model being called;
- the document list on the left shows short names, and the full title on hover;
- Stop ends an answer.

If the answer model writes citations that "could not be checked", read `HANDOVER.md`,
"Where it is most likely to need fixing", item 1.

Then `python -m docqa.stats`. These numbers are the baseline. Write them down.

### 7. Speed settings, one at a time

Work down the list in `HANDOVER.md`, "Making it faster", section "To try". For each:
change the setting, restart the screen, ask the same three questions, record the times
and whether the answers and citations are still right. Ask the user before keeping a
change to a number that `CLAUDE.md` names.

After switching `reranker.quantize`, the threshold has to be chosen again. After
switching `embedding.quantize`, run step 4 again: every document is embedded again.

### 8. Leave the project in a state the next person can trust

- Put the measurements in `machine.md`, section "Laptop A": the baseline and each
  setting tried, with the times.
- Update `HANDOVER.md`: in the table of "Where things stand", move what you saw working
  out of the column "Never checked".
- Commit on the branch, in small commits that say what changed and why.

## Your final report

1. The baseline: the time of each step for the three questions.
2. Each setting tried: before, after, and whether it was kept.
3. What failed and how it was fixed, with the files changed.
4. What you could not test.
5. What the user has to decide.

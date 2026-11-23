# Task for GitHub Copilot: bring this copy up to release 2026-11-23b and measure it

How to use this file: open the project folder in VS Code, open Copilot Chat in agent
mode, and say: "Read COPILOT_UPDATE.md and carry it out, one step at a time."

## The situation

This machine has the models (Ollama with `qwen2.5:7b` and `qwen2.5vl:7b`, BGE-M3, the
reranker, Docling's models) and a working copy of release 2026-11-23, with two commits
of its own on top (`--ingested-only` for a folder outside the project, and lower speed
settings). It has 15 PDFs ingested. Answers are still too slow for the user, who asked
for three things:

1. an answer in at most two minutes;
2. a progress tracker while an answer is made;
3. adding PDFs from the chat screen.

Release 2026-11-23b does these. It was written on a laptop that has none of the models,
so **none of its changes has run with a real model**. Its 574 fast tests pass there.
`HANDOVER.md`, section "What changed on 2026-11-23b", lists the changes. In short:

- `generation.time_limit_s: 120`. The evidence given to the answer model is cut to what
  it can read in the time left, at the speed measured on this machine, and the writing
  stops when the time is up;
- the reranker and the embedding model are run without a wasted trial run of the
  library, the reranker with 8-bit weights, and ranking has a time limit of its own;
- every answer shows its steps as they happen;
- "+ Add" in the side panel of the chat screen ingests PDFs;
- settings of this machine go in `config.local.yaml`.

Your job: get the release onto this machine, show that it works with the real models,
measure whether an answer now ends within two minutes, and report the numbers.

Two findings to keep in mind. The database is not what is slow: a search over the 79
indexed passages takes about 6 ms, so do not replace Qdrant or SQLite. And two minutes
with a 7B model on this processor leave room for one or two passages of evidence, not
six: that is the price of the limit, not a fault to repair.

## Read first

1. `AGENTS.md`: the rules you work under.
2. `HANDOVER.md`, sections "What changed on 2026-11-23b", "Making it faster", "Updating
   the copy on Laptop A" and "Where it is most likely to need fixing".
3. `CLAUDE.md`: the project's decisions. Where it says "ask me", ask the user.

## Rules that matter most

- **Nothing leaves the machine.** Do not set `llm.allow_remote` to true, and do not add
  a call to any hosted model or service, unless the user tells you to in this chat.
- **The stack is fixed.** Do not swap, add or upgrade a component, a model or a pinned
  version without asking. A smaller answer model is the user's decision, not yours.
- **One change at a time, measured.** Note the times before, change one setting, ask
  the same questions, note the times after.
- **Fix the cause.** Never weaken, skip or delete a test to make it pass.
- **Report plainly.** Say what you ran and what you saw. Say what you did not test.
- **The GitHub repository is public.** Commit code and notes only: no PDFs, nothing
  from `data/`, no keys, no query log, and not `config.local.yaml`.

## Stop and ask the user

- before changing a number that `CLAUDE.md` names or that this file does not tell you
  to change;
- before writing a new rerank threshold into `config.yaml`;
- before a run that is likely to take more than about 15 minutes;
- before ingesting documents that were not ingested before (step 8 needs one: ask
  which);
- before creating a model in Ollama (step 7);
- before pushing to GitHub.

## Steps

After each step, tell the user in two or three lines what you ran and what you saw.

### 1. Record where this machine stands

```
python -m docqa.stats
git status
git log --oneline -4
```

Write the statistics down: they are the "before". The log should show "Reduce retrieval
and answer latency" on top. Commit anything `git status` lists as changed.

### 2. Get the release

Stop the chat screen. Then follow `HANDOVER.md`, "Updating the copy on Laptop A": from
the bundle file, from GitHub, or from the zip, whichever the user gave you. With the
bundle it is one command, on the branch `speed-and-new-ui`:

```
git pull "<path to>\docqa-update-2026-11-23b.bundle" speed-and-new-ui
```

Afterwards `git log --oneline -2` should show the release commit on top of "Reduce
retrieval and answer latency".

### 3. Say where the documents are

Create `config.local.yaml` in the project folder:

```yaml
paths:
  docs_dir: "D:/EY Internship/docqa-docs-rbi-public-2026-11-20/docqa/docs"
```

Check the path first. If `git diff` in step 1 showed settings in `config.yaml` that
are particular to this machine, put them in this file instead. `config.local.yaml` is
not in git.

### 4. Environment and fast tests

```
python -m pip install --no-deps -e .
python -m docqa.check_env
pytest -m "not slow"
```

Every `check_env` line should say PASS, its notes should name the documents folder with
about 600 PDFs, and 574 tests should pass.

### 5. The real-model tests that are new

```
pytest -m slow -k "run_directly or instructions_twice or time_limit or added_from_the_chat_screen" -s
```

Run them one at a time if one fails. They were never run: a failure may be a mistake in
the test as easily as in the code. They print the numbers this release is about. Note
all of them:

- the reranker on 15 passages: run directly, through the library's call, and with
  8-bit weights (three times);
- the embedding model on 5 chunks: run directly and through the library's call;
- the answer model's instructions, read for the first time and again;
- two answers through the chat screen's server, with the time of every step.

If a `run_directly` test fails because the numbers differ from the library's, that is a
fault in this release: report it, with the numbers. `HANDOVER.md`, "Where it is most
likely to need fixing", item b, says how to switch the direct runs off meanwhile.

### 6. The screen: is an answer ready in two minutes?

```
python -m docqa.web --open
```

Wait for "Ready". Ask the question that took 329 s, and two more: one that a table
answers, and one that the documents cannot answer. Ask each of the first two twice. For
every answer note what the tracker shows: the time of each of the five steps, the total
against the limit ("1:48 of 2:00"), how many candidates were ranked ("15" or "9 of
15"), and how many passages and characters the answer model was given. Check:

- the steps light up one after the other, and the bar fills as time passes;
- the passages appear before the answer starts, and the answer arrives word by word;
- each citation is a document title with a page, and clicking it opens the passage;
- an answer that was stopped ends with a complete statement and says that it was
  stopped;
- the unanswerable question is refused, with "Reading" and "Writing" marked as not
  needed;
- Stop ends an answer.

Then `python -m docqa.stats`. Compare with step 1. Report to the user, for the second
asking of each question: the total, and whether it was within 120 s. If answers still
run over, `HANDOVER.md`, "Where it is most likely to need fixing", item a.

### 7. How many threads Ollama uses

This is an experiment the user has to agree to, because it creates a model name in
Ollama. `HANDOVER.md`, "Making it faster", item 1, has the commands: in
`ollama run qwen2.5:7b --verbose`, compare "prompt eval rate" and "eval rate" for
`num_thread` 2, 4, 6, 8 and 10 on the same input. Report the table. If one value is
clearly better than what Ollama does by itself, ask the user whether to make the model
`qwen2.5-7b-threads` with it and to name it in `config.local.yaml`, then repeat step 6.

### 8. Add a document from the screen

Ask the user which document. In the chat screen press "+ Add", tick it (or drop the
file on the dialog), leave "Describe figures" as the user wants it, and press Ingest.
Watch the run and Task Manager's memory graph. Check:

- each file shows its step (reading, describing figures, embedding and storing);
- the document appears in the list on the left when the run ends, and a question about
  it is answered from it;
- the question box is closed while the run goes on and open again afterwards;
- Stop ends a run after the step in hand (try it on a second document if the user
  agrees).

### 9. Leave the project in a state the next person can trust

- Put the measurements in `machine.md`, section "Laptop A": the numbers of steps 5, 6
  and 7, with the date.
- Update `HANDOVER.md`: in the table of "Where things stand", move what you saw working
  out of the column "Never checked".
- Commit on the branch, in small commits that say what changed and why.

## Your final report

1. Before and after: the statistics of step 1 and of step 6.
2. For each of the three questions: the time of each step, the total, and whether it
   was within two minutes.
3. The numbers the tests of step 5 printed.
4. The thread experiment, if it was done.
5. What failed and how it was fixed, with the files changed.
6. What you could not test.
7. What the user has to decide: the threshold (it was never chosen from real
   questions, and the reranker now runs with 8-bit weights), and whether two minutes
   with one or two passages of evidence answer well enough, or whether a faster model
   server is needed (`HANDOVER.md`, "What settings cannot do").

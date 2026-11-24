# Task for GitHub Copilot: bring this copy up to release 2026-11-24 and measure it

How to use this file: open the project folder in VS Code, open Copilot Chat in agent
mode, and say: "Read COPILOT_UPDATE.md and carry it out, one step at a time."

## The situation

This machine has the models (Ollama with `qwen2.5:7b` and `qwen2.5vl:7b`, BGE-M3, the
reranker, Docling's models) and a working copy of release 2026-11-23, with two commits
of its own on top. It has 15 PDFs ingested. Answers are too slow for the user.

Release 2026-11-24 was written on a laptop that has none of the models, so **none of
its changes has run with a real model**. Its 555 fast tests pass there. `HANDOVER.md`,
section "What changed on 2026-11-23b and 2026-11-24", lists the changes. In short:

- the reranker and the embedding model are run without a wasted trial run of the
  library, and the reranker with 8-bit weights;
- the answer model is given its instructions when the screen starts;
- every answer shows its steps as they happen;
- "+ Add" in the side panel of the chat screen ingests PDFs;
- settings of this machine go in `config.local.yaml`;
- another answer model can be tried and measured without changing the code.

**There is no time limit on an answer, and there must not be one.** A release in
between (2026-11-23b) limited an answer to two minutes by giving the answer model less
evidence. The user rejected that: it costs answers. Do not cut evidence, candidates or
answers to save time unless the user asks for exactly that.

Your job: get the release onto this machine, show that it works with the real models,
and measure where the time of an answer goes now. If the user agrees, try the smaller
answer model that `notes/2026-11-24-faster-answers-without-limits.md` recommends and
report how its answers and its times compare.

One finding to keep in mind. The database is not what is slow: a search over the 79
indexed passages takes about 6 ms. Most of an answer is the 7B model reading the
passages on the processor. Do not replace Qdrant or SQLite.

## Read first

1. `AGENTS.md`: the rules you work under.
2. `HANDOVER.md`, sections "What changed on 2026-11-23b and 2026-11-24", "Making it
   faster", "Updating the copy on Laptop A" and "Where it is most likely to need
   fixing".
3. `notes/2026-11-24-faster-answers-without-limits.md`: what was looked at, and why.
4. `CLAUDE.md`: the project's decisions. Where it says "ask me", ask the user.

## Rules that matter most

- **Nothing leaves the machine.** Do not set `llm.allow_remote` to true, and do not add
  a call to any hosted model or service, unless the user tells you to in this chat.
- **The stack is fixed.** Do not swap, add or upgrade a component, a model or a pinned
  version without asking. Downloading a model that `CLAUDE.md` does not list needs the
  user's yes in this chat, each time.
- **No time limits, and nothing that gives a model less to read to save time,** unless
  the user asks for it.
- **One change at a time, measured.** Note the times before, change one thing, ask the
  same questions, note the times after.
- **Fix the cause.** Never weaken, skip or delete a test to make it pass.
- **Report plainly.** Say what you ran and what you saw. Say what you did not test.
- **The GitHub repository is public.** Commit code and notes only: no PDFs, nothing
  from `data/`, no keys, no query log, and not `config.local.yaml`.

## Stop and ask the user

- before downloading or creating any model (steps 7 and 8);
- before changing a number that `CLAUDE.md` names;
- before writing a new rerank threshold into `config.yaml`;
- before a run that is likely to take more than about 15 minutes;
- before ingesting documents that were not ingested before (step 9 needs one: ask
  which);
- before pushing to GitHub.

## Steps

After each step, tell the user in two or three lines what you ran and what you saw.

### 1. Record where this machine stands

```
python -m docqa.stats
git status
git log --oneline -4
```

Write the statistics down: they are the "before". Commit anything `git status` lists
as changed.

### 2. Get the release

Stop the chat screen. Then follow `HANDOVER.md`, "Updating the copy on Laptop A": from
the bundle file, from GitHub, or from the zip, whichever the user gave you. With the
bundle it is one command, on the branch `speed-and-new-ui`:

```
git pull "<path to>\docqa-update-2026-11-24.bundle" speed-and-new-ui
```

It fits whether or not the bundle of 2026-11-23b was pulled before. Afterwards
`git log --oneline -3` should show "Release 2026-11-24" on top.

### 3. Say where the documents are

Create `config.local.yaml` in the project folder, unless it is there already:

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

Every `check_env` line should say PASS, its note should name the documents folder with
about 600 PDFs, and 555 tests should pass.

### 5. The real-model tests that are new

```
pytest -m slow -k "run_directly or instructions_twice or every_step or added_from_the_chat_screen" -s
```

Run them one at a time if one fails. They were never run: a failure may be a mistake in
the test as easily as in the code. They print numbers. Note all of them:

- the reranker on 15 passages: run directly, through the library's call, and with
  8-bit weights (three times);
- the embedding model on 5 chunks: run directly and through the library's call;
- the answer model's instructions, read for the first time and again;
- two answers through the chat screen's server, with the time of every step.

If a `run_directly` test fails because the numbers differ from the library's, that is a
fault in this release: report it, with the numbers. `HANDOVER.md`, "Where it is most
likely to need fixing", item a, says how to switch the direct runs off meanwhile.

### 6. The screen, and where the time goes

```
python -m docqa.web --open
```

Wait for "Ready". Ask the question that took 329 s, and two more: one that a table
answers, and one that the documents cannot answer. Ask each of the first two twice. For
every answer note what the tracker shows: the time of each of the five steps, the
total, how many passages and characters the answer model was given, and what the
tracker said the reading would take. Check:

- the steps light up one after the other, and the bar moves;
- the passages appear before the answer starts, and the answer arrives word by word;
- each citation is a document title with a page, and clicking it opens the passage;
- every answer is written to its end, from all the passages marked as given;
- the unanswerable question is refused, with "Reading" and "Writing" marked as not
  needed;
- Stop ends an answer.

Then `python -m docqa.stats`. Compare with step 1 and report: the total of a typical
answer before and after, and how it divides into ranking, reading and writing.

### 7. How many threads Ollama uses

Ask the user first: this may end in creating a model name in Ollama. `HANDOVER.md`,
"Making it faster", item 2, has the commands: in `ollama run qwen2.5:7b --verbose`,
compare "prompt eval rate" and "eval rate" for `num_thread` 2, 4, 6, 8 and 10 on the
same input. Report the table. If one value is clearly better than what Ollama does by
itself, ask the user whether to make the model `qwen2.5-7b-threads` with it and to
name it in `config.local.yaml`, then repeat step 6.

### 8. A smaller answer model, if the user agrees

`notes/2026-11-24-faster-answers-without-limits.md` recommends trying
`qwen3:4b-instruct-2507-q4_K_M`: about half the size, so about half the reading and
writing time. `CLAUDE.md` does not list it, so **ask the user before downloading it**,
and tell them to check its licence on its model page first (Apache 2.0 according to
search results).

If they agree:

```
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

If `eval/questions.csv` has questions, run them with both models and put the two
`results.csv` side by side:

```
python -m docqa.evaluate
python -m docqa.evaluate --answer-model qwen3:4b-instruct-2507-q4_K_M
```

Otherwise name the model in `config.local.yaml` (`llm: answer_model: ...`), restart
the screen and ask the questions of step 6 again. `python -m docqa.stats` then shows
both models under "By answer model". Report, for each question: both answers, whether
each statement has a citation that could be checked, whether figures are quoted
exactly, and the times. Do not judge which model is better: show the user and let
them decide. Put `llm.answer_model` back unless the user says to keep the new model.

### 9. Add a document from the screen

Ask the user which document. In the chat screen press "+ Add", tick it (or drop the
file on the dialog), leave "Describe figures" as the user wants it, and press Ingest.
Watch the run and Task Manager's memory graph. Check:

- each file shows its step (reading, describing figures, embedding and storing);
- the document appears in the list on the left when the run ends, and a question about
  it is answered from it;
- the question box is closed while the run goes on and open again afterwards;
- Stop ends a run after the step in hand (try it on a second document if the user
  agrees).

### 10. Leave the project in a state the next person can trust

- Put the measurements in `machine.md`, section "Laptop A": the numbers of steps 5 to
  8, with the date.
- Update `HANDOVER.md`: in the table of "Where things stand", move what you saw working
  out of the column "Never checked".
- Commit on the branch, in small commits that say what changed and why.

## Your final report

1. Before and after: the statistics of step 1 and of step 6.
2. For each of the three questions: the time of each step and the total.
3. The numbers the tests of step 5 printed.
4. The thread experiment, if it was done.
5. The smaller answer model, if it was tried: answers and times next to the 7B's.
6. What failed and how it was fixed, with the files changed.
7. What you could not test.
8. What the user has to decide: which answer model to use, and the threshold (it was
   never chosen from real questions, and the reranker now runs with 8-bit weights).

# Task for GitHub Copilot: make this build run with the real models

**How to use this file.** Open the project folder in VS Code, open Copilot Chat, choose
Agent, and send this one message:

```text
Read COPILOT_PROMPT.md and carry it out from step 1. Stop wherever it tells you to stop.
```

Everything below the line is addressed to the assistant.

---

## The situation

This repository is a local question-answering chatbot for banking documents. It parses
PDFs, indexes them, searches them and writes answers that cite the document and page.

All of the code is written and its 347 fast tests pass. **It has never been run with a
real model.** It was written on a laptop with no Ollama, no embedding model, no
reranker and none of Docling's models, so every model was replaced by a small fake in
the tests. This machine has Ollama with both Qwen models. Your job is to connect the
code to the real models, find what breaks, fix it, and show that the chatbot answers
real questions with correct citations.

Expect real problems. The parts that call a model were written from the libraries'
signatures, not by running them. `HANDOVER.md`, section "Where it is most likely to
need fixing", lists the weak points in order of likelihood. Start your thinking there
when something fails.

## Read first

1. `AGENTS.md`: the rules you work under.
2. `CLAUDE.md`: the project's decisions. Where it says "ask me", ask the user.
3. `HANDOVER.md`: what exists, what was and was not checked, and the setup commands.
4. `README.md`: the commands.

`design.md` explains the methods in detail. Read a section when you work on that part.

## Rules that matter most

- **The stack is fixed.** Docling, Ollama, `qwen2.5:7b`, `qwen2.5vl:7b`, BGE-M3,
  `bge-reranker-v2-m3`, Qdrant in local mode, SQLite, Streamlit. Do not swap, add or
  upgrade a component, and do not change a pinned version, without asking the user.
- **Nothing leaves the machine.** No cloud AI service may be called by the pipeline.
  The only network use is installing the pinned packages and downloading the models
  named in step 4.
- **Fix the cause.** When a test fails, find out whether the code or the test is
  wrong. Never weaken, skip or delete a test to make it pass. If a test is wrong, say
  what was wrong with it.
- **Settings live in `config.yaml`.** No model name, path, size or threshold in code.
- **Fast tests stay model-free.** Anything that needs a real model is marked
  `@pytest.mark.slow`. When a real run exposes a bug that the fakes hid, fix the code
  and add a fast test that would have caught it.
- **Report plainly.** Say what you ran and what you saw. Do not call something working
  unless you ran it and saw it work. Say what you skipped and what you could not test.
- **Small commits on a branch**, each with a message that says what changed and why.
  Do not commit `data/`, `models/` or the PDFs.

## Stop and ask the user

- before downloading any model (step 4);
- before installing anything that is not in `requirements.txt`;
- if a pinned package cannot be installed on this machine's Python;
- before a run that is likely to take more than about 15 minutes;
- before changing the stack, a pinned version, or a rule in `CLAUDE.md`;
- which documents to ingest (step 7), and for every expected answer (step 8);
- before writing a rerank threshold into `config.yaml` (step 9);
- if a file looks as if it holds real customer data.

## Steps

Do them in order. After each one, tell the user in two or three lines what you ran and
what you saw.

### 1. Describe this machine

`machine.md` describes the laptop the code was written on. Rewrite it for this machine:
hardware, operating system, Python version, Ollama version and the models it lists,
and anything about the network that matters. Keep the section "Behaviour of installed
software found during setup": it records traps that still apply.

### 2. Python environment

Follow `HANDOVER.md`, "Python environment". If pip cannot install an exact pin for this
Python version, stop and ask. Do not loosen a pin yourself.

### 3. Ollama

Confirm that `ollama list` shows `qwen2.5:7b` and `qwen2.5vl:7b`. Set
`OLLAMA_CONTEXT_LENGTH` and `OLLAMA_NO_CLOUD` as `HANDOVER.md` describes, restart
Ollama, and confirm with `ollama ps` after one request that the context length took
effect. If the variable names no longer work in this Ollama version, find the current
ones in Ollama's own documentation and tell the user what changed.

### 4. Download the remaining models

There are three downloads: BGE-M3, the reranker, and Docling's models. For each one,
before downloading, tell the user its name, where it comes from, its size, and the
licence stated on its model page. `CLAUDE.md` allows only licences that permit
commercial use. Wait for a yes. Then run the commands in `HANDOVER.md`, "Download the
other models, once", and record the licences in `machine.md`.

### 5. Environment check and fast tests

Run `python -m docqa.check_env` until every line says PASS. Then run
`pytest -m "not slow"`. All 347 should pass on this machine too. A failure here is most
likely a difference between machines, such as a path or a package version: fix that,
not the test.

### 6. Real-model tests

Run the groups in `tests/test_real_models.py` one at a time, in the order given in
`HANDOVER.md`, "Check". These tests have never been run, so for every failure decide
which of the two is wrong, the code or the test, and say which.

Do not go on to step 7 until the last test in that file passes: a small PDF is
ingested with the real models, a question about it gets an answer that cites its page,
and an unrelated question is refused.

### 7. Ingest a few real documents

Ask the user which two or three PDFs in `docs/` to start with. Prefer short ones that
have a table. Ingest them with `python -m docqa.ingest` and then look at the results
yourself:

- `data/parsed/`: did Docling find the headings, lists and tables that the PDF has?
- the chunks: are they within `chunking.min_tokens` and `chunking.max_tokens` when
  counted with the real tokenizer? Does every chunk have the right page?
- `data/figures/`: are the crops real figures, or logos and stamps?
- the warnings and timings the command printed.

Fix what is wrong in `src/docqa/ingest/`. Record the seconds per step in `machine.md`,
replacing the estimates in `design.md` section 15 with measurements.

### 8. Test questions

`eval/questions.csv` has only its header row. With the user, write 10 to 20 questions
about the ingested documents, including at least five that no document answers (type
`not_in_docs`). `eval/README.md` explains the columns. Do not invent an expected
answer: open the page, quote it, and have the user confirm it. A wrong expected answer
makes every later number meaningless.

### 9. Rerank threshold

Run `python -m docqa.evaluate --suggest-threshold`. Show the user the best score of
each question, from `results.csv`, and the value the script proposes. If the two
groups of questions overlap, say so: no threshold separates them, and the search needs
attention before a threshold does. With the user's agreement, set
`retrieval.rerank_score_threshold` in `config.yaml`.

### 10. Full evaluation

Run `python -m docqa.evaluate`. Report the summary table. For every question that went
wrong, use `results.csv` and the query log to say which of these it was:

- the search did not find the right page;
- the right page was found and the model answered wrongly;
- the answer was right and the citation was wrong or could not be matched;
- it refused when it should have answered, or answered when it should have refused.

Fix what you can within the rules above, and run the evaluation again. Raise
`PROMPT_VERSION` in `src/docqa/generate/prompt.py` whenever you change the
instructions to the answer model.

### 11. Chat screen

Start `streamlit run src/docqa/app/main.py`. Ask three questions: one answered from
text, one from a table, and one that should be refused. Check that the answer streams,
that the citations are listed, that each source opens, that a table is shown as a
table, and that the question appears in the query log (`data/docqa.sqlite`, table
`query_log`). Stop the app before running ingestion or evaluation again.

### 12. Figures and scanned pages

Only once steps 1 to 11 work for text and tables. Ingest one document with a chart or
flowchart and one with a scanned page. Read what the vision model wrote about each
figure next to the figure itself, and what OCR made of the scanned page. Report
honestly how good they are. Add questions of type `chart`, `flowchart` and `scanned`
to the question file and run the evaluation again.

### 13. Leave the project in a state the next person can trust

- Update the status table in `HANDOVER.md`: what is now checked with real models.
- Update the commands in `CLAUDE.md` and `README.md` if any changed.
- Make sure `pytest -m "not slow"` and `pytest -m slow` both pass.
- Commit.

## Your final report

Give the user:

- what works, with the command you ran and what you saw for each claim;
- the evaluation table per question type, and the threshold in use;
- what does not work yet, and what you tried;
- what you did not test;
- every change you made to the code, in one line each, and every place where you
  changed a test, with the reason.

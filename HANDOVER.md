# Handover: setting up, updating and speeding up the build

Written on 2026-11-20. Updated on 2026-11-23 with releases 2026-11-23 and 2026-11-23b,
and on 2026-11-24 with release 2026-11-24.

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
  whole answer.

**Release 2026-11-23 ran on Laptop A on 2026-11-23** (`machine.md` has the record):
`check_env` passed all 24 checks, bringing the 15 documents up to date took 404 s
(384 s of it embedding and storing), and the four new real-model tests passed. The
8-bit option was timed there on short texts: 12 passages ranked in 1.5 s instead of
3.1 s, 12 texts embedded in 1.2 s instead of 1.8 s. Two commits were made on that
laptop and are part of this release: `--ingested-only` now matches documents in a
folder outside the project, and the speed settings were lowered (15 candidates, 4
passages kept, 8,000 characters of evidence, 400 tokens). Answers were still found
too slow. How long they took with those settings was not recorded.

**Release 2026-11-24, this one, has not been run with a real model.** It is release
2026-11-23b without its time limits. Release 2026-11-23b was asked for on 2026-11-23:
an answer in at most two minutes, a progress tracker on the answer, and adding PDFs
from the chat screen. The two minutes could only be kept by giving the answer model
one or two passages to read where four were found. The project owner did not want
that, so on 2026-11-24 the limits were taken out, and what makes an answer faster
without giving the models less to read was looked into instead
(`notes/2026-11-24-faster-answers-without-limits.md`). It is not recorded whether
release 2026-11-23b was installed on Laptop A in between. Everything in the table
marked "stand-ins" was tested with fakes in place of the models.

| Part | Code | How it was checked | Never checked |
|---|---|---|---|
| Configuration, environment check | `config.py`, `check_env.py` | Tests; `check_env` run for real on both laptops (Laptop A: 24 passed) | `config.local.yaml` on Laptop A |
| Document identity and manifest | `ingest/documents.py`, `scripts/build_manifest.py` | Tests; the manifest was built from the real RBI index and matched 607 of the 618 PDFs | |
| Docling parsing | `ingest/parser.py` | Tests of the options and the parse cache. 15 real PDFs on Laptop A; parsing without OCR passed its real-model test there | |
| Text-layer check (`parsing.ocr: auto`) | `ingest/parser.py` | Tests; run over all 618 real PDFs on Laptop B: 7 have a scanned page | |
| Reading Docling's output | `ingest/elements.py`, `ingest/tables.py` | Tests on Docling documents built by hand. 15 real PDFs on Laptop A | Which real pictures the size rules skip |
| Personal-data check | `ingest/screening.py` | Tests | Its false-alarm rate on real circulars |
| Figure descriptions | `ingest/figures.py` | Tests with a faked model server. 3 real figures on Laptop A | Descriptions of the scaled-down images |
| Chunking | `ingest/chunker.py` | Tests with words counted as tokens | Sizes with BGE-M3's real tokenizer |
| Ingestion in steps | `ingest/pipeline.py` | Tests with stand-ins; the refresh of 15 documents on Laptop A | Stopping a run, and a chosen set of files, with real models |
| Embedding and reranking, run directly | `index/embedder.py`, `retrieve/reranker.py`, `batching.py` | Tests against the library itself on a small model of the same kind: the same numbers, one model run fewer. Timed on Laptop B with a stand-in of the reranker's size | **With the real models.** Three slow tests do it |
| 8-bit weights | `quantize.py` | Laptop A's real-model tests, on short texts | How it changes which passages are found on real questions |
| Vector store | `index/store.py` | Tests against the real Qdrant client; timed with 79, 3,000 and 20,000 passages (`machine.md`) | |
| Retrieval, reporting each step as it happens | `retrieve/__init__.py` | Tests: real Qdrant, stand-in models | Search quality |
| Answers and citations | `generate/` | Tests with a faked model server. Real answers on Laptop A with release 2026-11-23 | |
| Trying another answer model | `evaluate.py` (`--answer-model`), `llm.py` (`llm.reasoning_effort`), `stats.py` | Tests with a faked model server | With a second real model |
| Chat screen and its progress tracker | `web/server.py`, `web/static/` | Tests through the server's own addresses; opened in a browser on Laptop B with stand-in models | With real answers |
| Adding documents from the screen | `web/ingestion.py` | Tests through the server's own addresses with the real pipeline, register and index; done in a browser on Laptop B with a stand-in for Docling | With Docling and the real models; memory use while it runs |
| Statistics | `stats.py`, `db.py` | Tests | The new figures on a real query log |
| Evaluation | `evaluate.py` | Tests with stand-ins | On real questions. `eval/questions.csv` has no rows yet |
| Comparing an older copy | `scripts/compare_with_release.py` | Tests; run on the 2026-11-20 zip and on the code from Laptop A | |

Test counts on 2026-11-24: 555 fast tests pass on Laptop B (`pytest -m "not slow"`). 21
tests need the real models (`pytest -m slow`, in `tests/test_real_models.py`). Laptop A
ran four of them. The six of group 8 are new since release 2026-11-23 and have been
run nowhere. A failure there may be a mistake in the test as easily as in the code.

## What changed on 2026-11-23b and 2026-11-24

Release 2026-11-23b also gave an answer a time limit of 120 seconds and ranking one of
20 seconds. **Both limits were taken out on 2026-11-24.** An answer is written from
every passage that scores well enough, every candidate is ranked, and the model writes
to its end. What is listed here is what stayed and what was added on 2026-11-24.

1. **Ranking does a quarter less work.** The FlagEmbedding library runs its longest
   batch of passages once as a trial, throws the result away and runs it again. The
   reranker is now run from the project's own loop (`retrieve/reranker.py`), with the
   inputs prepared by the library's own functions, so the scores are the same. Timed
   on Laptop B with a stand-in model of the reranker's size, for 15 passages: 42 s the
   library's way, 33 s without the trial run. With 8-bit weights: 22 s and 14 s.
2. **The reranker runs with 8-bit weights:** `reranker.quantize: true`. Laptop A's own
   test showed it ranking twice as fast with the answering passage still first. It is
   the one speed setting that is on and can change a ranking: what it does to real
   questions has not been measured. `false` gives full precision at about twice the
   time. The threshold of 0.50 was a temporary value before and is one now.
3. **The embedding model is run the same way.** Ingestion embeds one document at a
   time. A document of five chunks was three runs of the model; it is two now.
4. **The answer model is given its instructions when the screen starts.** Ollama keeps
   what it has read of the last request, and every question starts with the same
   instructions, so the first question does not wait for them to be read.
5. **Every answer shows a progress tracker:** five steps (question, search, ranking,
   reading, writing), the one in hand with a running clock, and a bar for how far that
   step is: exact for ranking, which counts its passages, and by the answer model's
   measured reading speed for reading. The server sends each step as it happens and
   the page reads them with the browser's Streams API. The estimate is for the reader:
   nothing is cut or stopped because of it.
6. **PDFs can be added from the chat screen:** "+ Add" in the side panel. Files are
   dropped or chosen there and copied to `data/uploads`, or ticked in the list of PDFs
   that lie in the documents folder and were never ingested. The run shows each file
   and each step, can be stopped, and refuses a file that looks like personal data, as
   the command line does. Questions wait while it runs. The screen no longer has to be
   stopped to ingest.
7. **Settings of one machine go in `config.local.yaml`,** next to `config.yaml`, with
   only the keys that differ. It is not in git, so an update of the project does not
   overwrite it and it never reaches the other laptop. Laptop A needs it for the folder
   its documents are in.
8. **A document keeps its name whichever part of the documents folder is ingested.**
   The fix made on Laptop A named a file by its place below the folder given on the
   command line. For a sub-folder of the documents folder that gave a known document a
   second name. Inside the documents folder the name is now always the place below
   that folder; a folder elsewhere is named as on Laptop A. `--prune` no longer
   removes documents whose files are in another of the folders.
9. **Another answer model can be tried and measured without changing the code**
   (2026-11-24): `python -m docqa.evaluate --answer-model NAME` runs the test questions
   with it, `python -m docqa.stats` shows each answer model's times side by side, and
   `llm.reasoning_effort` switches off the thinking of a model that thinks before it
   answers.
10. **The statistics show** the ranking time and the answer model's reading speed.
11. **A request that another website makes the browser send is refused** (asking,
    uploading, ingesting).

No document has to be ingested again for this release.

## What changed on 2026-11-23

1. **The chat screen is the project's own web page**, not Streamlit:
   `python -m docqa.web --open`. On the left it lists the ingested documents by a short
   name (point at one for the full title, file, version, date, pages and passages) and
   shows the figures: documents, pages, passages, questions asked, typical times. The
   passages found for a question appear before the answer starts. Stop ends an answer.
2. **Models are loaded when the screen starts** (`ui.warm_up`), in the background, so
   the first question does not pay for loading them.
3. **The answer model reads less.** Of the passages kept, only those that score at
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
9. **Optional 8-bit models** (`embedding.quantize`, `reranker.quantize`).
10. **The vision model can have its own server address** (`llm.vision_base_url`), and
    the screen and `check_env` say so when a model runs on another machine.
11. **`python -m docqa.ingest --ingested-only`** brings the documents already in the
    index up to date and leaves new files alone.
12. **Streamlit is no longer installed.** `starlette`, `uvicorn`, `pypdfium2`, `pillow`
    and `psutil` are pinned instead.

## Where the time of an answer goes

Measured on Laptop A with release 2026-11-20, for one question: the search took 81 s
and the whole answer 329 s. So the answer model took about 248 s. With the settings
lowered on 2026-11-23 answers were still found too slow; no times were recorded.

**The database is not the slow part.** Timed on Laptop B on 2026-11-23, with no model
loaded, through the project's own search code:

| Passages in the index | One database search | Share of the 81 s |
|---|---|---|
| 79 (what Laptop A has) | 6 ms | less than a hundredth of one percent |
| 3,000 | 0.2 s | |
| 20,000 | 1.6 s | |

Writing a question to the SQLite log took 14 ms. `machine.md` has the details.

**The search time is the reranker.** It reads the question against every candidate
passage, and it is a model of 568 million numbers on a laptop processor. Timed on
Laptop B (4 cores) with a model of the same size and shape, filled with random numbers,
for 15 passages of 180 to 820 tokens:

| How the 15 passages are scored | Full precision | 8-bit weights |
|---|---|---|
| The library's call, as releases up to 2026-11-23 did it | 42 s | 22 s |
| The project's own loop: sorted by length, no trial run (this release) | 33 s | 14 s |
| Batches in the order the search returned them | 41 s | 21 s |
| One passage at a time | 39 s | 15 s |

So this release should rank in about a third of the time that release 2026-11-23 took
with the same number of candidates. Laptop A's processor is a different one: its times
will differ, the proportions should not by much. Its real-model tests print them.

**The rest is the answer model,** reading and then writing. On a laptop processor a 7B
model reads some tens of characters a second and writes a few tokens a second. Since
release 2026-11-23 both are recorded with every answer, and `python -m docqa.stats`
prints the medians: "wait for the first word", "reading speed" and "writing speed".

**Why MongoDB or Atlas would not make it faster.** A different database replaces the
6 ms, not the minutes. It would also break two rules of the project. Atlas is a cloud
service, so the text of the documents would be stored outside the machine. MongoDB
itself is a server program to install and run, where Qdrant's local mode and SQLite are
libraries inside Python. If the index grows towards 20,000 passages the search does get
slower, and the step for that is already decided in `CLAUDE.md`: a Qdrant server,
through `qdrant.url`.

## Making it faster

**There is no limit on the time of an answer.** A limit was built on 2026-11-23 and
taken out the next day, because it could only be kept by giving the models less to
read. What follows makes an answer faster without that, or says plainly what it costs.
`notes/2026-11-24-faster-answers-without-limits.md` has the research behind it, with
its sources.

**Where the time goes now** (estimates for Laptop A; its own figures are in
`python -m docqa.stats`):

| Step | Time | What decides it |
|---|---|---|
| Question and database | under a second | |
| Ranking 15 passages | 15 to 25 s | the reranker, on the processor |
| The answer model reads | about 2 minutes | about 2,000 tokens of evidence at about 15 tokens a second |
| The answer model writes | 20 to 50 s | 100 to 200 tokens at 4 to 5 tokens a second |

About three minutes, two of them the 7B model reading. So the step that matters is the
reading, and the way to make it faster without less evidence is a faster answer model.

**Measure first.** Ask the same three questions before and after each change. The
tracker on each answer shows the time of every step, and this prints the medians, for
all answers and for each answer model on its own:

```powershell
python -m docqa.stats
```

**Switched on:**

| What | Setting | What it costs |
|---|---|---|
| Ranking without the library's trial run | none (`retrieve/reranker.py`) | Nothing that is known: the scores are the same |
| 8-bit reranker | `reranker.quantize: true` | Scores shift slightly. Not measured on real questions |
| Instructions given at start | `ui.warm_up: true` | Nothing |
| From Laptop A, 2026-11-23 | `candidates: 15`, `top_k: 4`, `evidence_char_budget: 8000`, `max_tokens: 400` | Fewer passages found and given |
| From Laptop A, 2026-11-23 | Ollama keeps a model in memory for 30 minutes: the user environment variable `OLLAMA_KEEP_ALIVE=30m` | Memory stays in use |

**To try on Laptop A, most promising first:**

1. **A smaller answer model of the same family.** `Qwen3-4B-Instruct-2507` is about
   half the size of `qwen2.5:7b`, so it should read and write in about half the time:
   an answer of three minutes in a minute and a half to two, from all the evidence. A
   2026 study that ran 17 small models as the answering model on a processor found it
   the best trade of quality against time, with scores close to the 8B model of its
   family at a fifth of the time. Whether it answers these documents as well as the 7B
   has to be seen on real questions. **This is the project owner's decision:**
   `CLAUDE.md` lists the models that may be used, and the model has to be downloaded
   (2.5 GB). Its licence is Apache 2.0 according to search results; check its page.

   ```powershell
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ```

   Compare the two on the test questions, without changing a setting:

   ```powershell
   python -m docqa.evaluate --answer-model qwen3:4b-instruct-2507-q4_K_M
   ```

   To use it in the chat screen, name it in `config.local.yaml`
   (`llm: answer_model: qwen3:4b-instruct-2507-q4_K_M`). Take a name with "instruct"
   in it. The other Qwen3 models think before they answer, which takes longer than
   the answer; for those, set `llm.reasoning_effort: none` as well.
2. **How many processor threads Ollama uses.** Free, and it changes nothing about the
   answers. Laptop A's processor has 2 fast cores and 8 slow ones. Ollama chooses a
   number of threads by itself, and reports for such processors go both ways. Find
   out what is fastest:

   ```powershell
   ollama run qwen2.5:7b --verbose
   ```

   Paste a paragraph of text and ask for a summary. After the reply it prints
   "prompt eval rate" (reading) and "eval rate" (writing), in tokens per second. Then
   type `/set parameter num_thread 4`, ask again, and compare. Try 2, 4, 6, 8 and 10.
   If one number is clearly best, make it permanent with a model of your own name that
   differs from `qwen2.5:7b` only in that setting. Put these two lines in a file
   called `Modelfile`:

   ```
   FROM qwen2.5:7b
   PARAMETER num_thread 8
   ```

   ```powershell
   ollama create qwen2.5-7b-threads -f Modelfile
   ```

   and name it in `config.local.yaml` (`llm: answer_model: qwen2.5-7b-threads`). It is
   the same model file, so nothing is downloaded. Ollama's documentation gives a
   Modelfile as the way to set such things for its OpenAI-compatible address, which
   is the one this project uses.
3. **A smaller reranker.** `Alibaba-NLP/gte-reranker-modernbert-base` is about a
   quarter of the size of the present one (Apache 2.0 according to its page), so
   ranking should take about a quarter of the time. Ranking is a small part of an
   answer, so this comes after 1. It is a setting (`reranker.model`) and a download,
   and the owner's decision. Another reranker has another scale of scores: choose the
   threshold again and judge the search again (`python -m docqa.evaluate
   --suggest-threshold`, `--retrieval-only`).
4. **8-bit embedding model:** `embedding.quantize: true`. It does little for a
   question. It shortens ingestion: embedding was 384 s of the 404 s that bringing 15
   documents up to date took. The vectors change, so every document is embedded again
   once, and the stored vectors and the question's must be made with the same setting.

**Faster, at a cost to what is found or read.** These are the owner's to weigh, with
test questions to judge them by:

- Fewer candidates (`retrieval.candidates`), fewer passages kept (`retrieval.top_k`),
  less evidence (`generation.evidence_char_budget`).
- Smaller passages (`chunking`): the model reads half as much if passages are half the
  size. It changes what is found, every document has to be ingested again, and
  `CLAUDE.md` fixes the sizes.
- Shorter reranker input (`reranker.max_length: 512`): the reranker then judges a long
  passage by its first part only. `config.py` refuses a value below
  `chunking.max_tokens` on purpose.

For ingestion:

1. Leave the box "Describe figures" unticked, or use `--no-figures`. Figures are then
   indexed by their captions.
2. `parsing.table_mode: fast` uses Docling's quicker table model. Tables with merged
   cells may come out worse.
3. `embedding.quantize: true`, as above.
4. `ingestion.group_files` sets how many files are read before the models change. A
   larger group means fewer model loads; a smaller one means results sooner.

**What settings cannot do.** A 7B model on this processor takes about two minutes to
read four passages. Answers in seconds need other hardware: a GPU server with vLLM,
which is the plan in `CLAUDE.md`, or a hosted endpoint, next section.

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

What it would change, as an estimate: the answer model's share of an answer would
become seconds. The search would stay as it is.

A GPU server of your own with vLLM is the other route, and the one `CLAUDE.md` plans.
It uses the same four settings, and nothing goes to a third party. vLLM serves one
model per server, which is what `vision_base_url` is for.

## Updating the copy on Laptop A

`COPILOT_UPDATE.md` is this section as a task for GitHub Copilot.

Stop the chat screen first. Then get the code, in one of three ways:

**a. From the git bundle** (a single file, `docqa-update-2026-11-24.bundle`). It holds
two commits on top of the last commit that Laptop A pushed to GitHub (`81bea90`,
"Reduce retrieval and answer latency"): release 2026-11-23b, and this release on top
of it. So it fits whether or not the earlier bundle was already pulled. In the project
folder on Laptop A:

```powershell
git status
```

It should say `On branch speed-and-new-ui`. Commit anything it lists as changed, then:

```powershell
git pull "<path to>\docqa-update-2026-11-24.bundle" speed-and-new-ui
```

**b. From GitHub**, if the branch has been pushed there from another machine:
`git pull origin speed-and-new-ui`.

**c. From the zip** (`docqa-code-2026-11-24.zip`), if the project folder is not a git
repository. Do not extract it over the old folder. Extract it to a new folder, run
`python scripts\compare_with_release.py "<old folder>"` there to see what the old copy
changed, move `data\` and `models\` from the old folder to the new one, and activate
the old virtual environment.

Then, whichever way was used:

**Say where the documents are.** The documents on Laptop A are in a folder outside the
project. Create `config.local.yaml` in the project folder with these two lines (forward
slashes, and the quotes, are needed):

```yaml
paths:
  docs_dir: "D:/EY Internship/docqa-docs-rbi-public-2026-11-20/docqa/docs"
```

From then on `python -m docqa.ingest` needs no folder, and "+ Add" in the chat screen
lists the PDFs of that folder that were never ingested. If `config.yaml` was edited on
that laptop for other settings of its own, move those to this file too.

```powershell
python -m pip install --no-deps -e .
```

```powershell
python -m docqa.check_env
```

It should pass all its checks and end with a note that names the documents folder
(about 600 PDFs).

```powershell
pytest -m "not slow"
```

555 should pass. Then the real-model tests that are new, which also print the times
that matter:

```powershell
pytest -m slow -k "run_directly or instructions_twice or every_step or added_from_the_chat_screen" -s
```

No document has to be ingested again. Start the screen:

```powershell
python -m docqa.web --open
```

Ask three questions and watch the tracker: it shows the time of every step. From the
second answer on it also says how long the reading can be expected to take. Then:

```powershell
python -m docqa.stats
```

and follow "Making it faster". To add a document, use "+ Add" in the side panel.

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

555 should pass. Then the real-model tests, one group at a time, in this order:

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

The same can be done from the chat screen (step 8): "+ Add" in its side panel lists
the PDFs of the documents folder that were never ingested, and takes files dropped on
it. If the documents are kept in a folder outside the project, name it in
`config.local.yaml` first ("Updating the copy on Laptop A" shows how).

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
running `python -m docqa.ingest` or `python -m docqa.evaluate` in a terminal. All three
use the same index, and Qdrant's local mode lets only one program open it. Documents
can be added without stopping it, with "+ Add" in its side panel: that ingestion runs
inside the screen's own program.

## Where it is most likely to need fixing

In rough order of likelihood. Each item names the file to look at.

**New since release 2026-11-23. None of it has met a real model.**

a. **The search models run from the project's own loop** (`retrieve/reranker.py`,
   `index/embedder.py`). If the library's functions are not where this release expects
   them, the log says "run through the library's own call from now on" once and
   everything works as before, only slower. The slow tests named `run_directly` compare
   the numbers with the library's on the real models. Should they differ, set
   `self.direct = False` in the two classes and report it.
b. **The answer model reading its instructions only once** (`web/server.py`,
   `AppState.warm_up`). This relies on Ollama keeping what it read of the previous
   request. The slow test named `instructions_twice` checks it by timing. If it fails,
   the first answer after a start is slower and nothing else is wrong.
c. **Ingestion inside the chat screen** (`web/ingestion.py`). Before a run the search
   models are dropped from memory and Ollama is left to swap the answer model for the
   vision model; afterwards the search models are loaded again. Watch Task Manager
   during the first run with figures. If memory runs short, untick "Describe figures".
   Docling is loaded inside the screen's program for the first time here.
d. **The tracker's estimate of the reading time** (`Database.read_speed` in `db.py`,
   `web/static/app.js`). It is the characters of evidence of an answer divided by its
   wait for the first word, as a median over the last 20 answers of the answer model.
   If the line "About 50 s on this computer" is far off, the model was not in memory
   at the time, or was reading its instructions again (item b). Nothing depends on the
   estimate but the bar.
e. **The 8-bit reranker and the threshold.** Scores shift a little, and the threshold of
   0.50 was never chosen from real questions. A question that is refused although its
   passage is in the documents, or answered from an unrelated passage, points here:
   write questions in `eval/questions.csv` and run
   `python -m docqa.evaluate --suggest-threshold`.

**From release 2026-11-23 and before.** Laptop A has run all of this once.

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
    top passages for a question may be near-identical ones. The document title
    is part of what is embedded, which should help; it has not been measured.

## Decisions taken while building

`design.md` section 16 listed these as open. They were settled as follows so that the
code could be written. Each is easy to change.

| Question | What the code does | Where |
|---|---|---|
| Where titles, versions and dates come from | `manifest.csv` in the project root, one row per file. Built from the RBI index: 607 documents have their full title and date; the 11 others use their file name | `paths.manifest`, `ingest/documents.py` |
| What identifies a document | Its place below the documents folder (`paths.docs_dir`), whichever part of that folder is ingested. A file added from the chat screen is `uploads/<file>`. A file in a folder elsewhere is named by its place below that folder. An edited file replaces its chunks; a renamed file is a new document | `ingest/documents.py` |
| Which page is cited | Chunks record where each later page begins, and the model is shown those places, so it can cite the page a fact is on | `Chunk.page_marks`, `generate/prompt.py` |
| How the model names a source | By a label, `E1`, `E2` and so on, with the page. The code replaces the label with the document's title | `generate/citations.py` |
| Which passages the model gets | Of those kept, those scoring at least half as well as the best, within the evidence budget | `generation.evidence_score_ratio`, `generation.evidence_char_budget` |
| How long an answer may take | As long as it takes. Limits of 120 s for an answer and 20 s for ranking were built on 2026-11-23 and taken out the next day: they could only be kept by giving the models less to read | `notes/2026-11-24-faster-answers-without-limits.md` |
| What is embedded | The document title, then the headings above the chunk, then its text | `index.searchable_text` |
| Scale of rerank scores | Normalised, 0 to 1. The threshold is on that scale | `reranker.normalize` |
| When OCR runs | Only for a document that has a page with an image and almost no text | `parsing.ocr`, `ingest/parser.py` |
| Decorative pictures | Skipped without a caption if they cover under 2% of the page, are under 60 pixels on a side, or are 8 times longer than wide. A figure the vision model calls a logo, photograph, signature or stamp is not indexed unless something readable is in it | `figures`, `ingest/elements.py`, `ingest/figures.py` |
| A figure that was not described | Indexed by its caption alone, if it has one | `ingest/chunker.py` |
| Real customer data | A file with 3 or more PAN-like or Aadhaar-like numbers is not ingested | `screening`, `ingest/screening.py` |
| A table too long for the models | Never split. Search uses its first 1,024 tokens; the answer model gets as much as fits the evidence budget, marked as cut | `embedding.max_length`, `generation.evidence_char_budget` |
| Table of contents | Not indexed | `ingest/elements.py` |
| More than one question at a time | The second waits for the first. The models use the whole processor anyway. A question also waits while documents are being ingested from the screen | `web/server.py` |
| Adding documents | From the chat screen, into `data/uploads`, or by ticking files of the documents folder. One run at a time, inside the screen's program | `web/ingestion.py` |
| Settings that differ on one machine | In `config.local.yaml`, laid over `config.yaml`, not in git | `config.py` |
| The chat screen | A page of plain HTML, CSS and JavaScript served by Starlette and uvicorn on this machine. It loads nothing from the internet | `web/` |

Still open:

- A table that continues on the next page arrives as two tables and stays two.
- Follow-up questions. Each question is answered on its own.
- Where `data/` lives when the project folder is synced by OneDrive.
- Which 10 to 20 documents form the prototype set.
- Whether answers may come from a hosted endpoint (see that section).
- Whether a smaller answer model may be used ("Making it faster", item 1).

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

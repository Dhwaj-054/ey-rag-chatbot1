# Machine profiles

Two laptops are used for this project. Read the section for the one you are working on.

| | Laptop A: runs the models | Laptop B: where the code is written |
|---|---|---|
| CPU | Intel Core i5-1335U, 10 cores, 12 logical | Intel Core i7-1185G7, 4 cores, 8 logical |
| RAM | 15.7 GB | 31.7 GB |
| GPU | Integrated only, so everything runs on the CPU | Integrated only |
| Windows | 11 Home | 11 Enterprise |
| Python | 3.13.3 | 3.14.6 |
| Ollama program | 0.40.16 | Not installed. Its download site is blocked there |
| BGE-M3, reranker, Docling's models | On the machine | Not on the machine. Hugging Face is blocked there |
| What can run | Everything | The fast tests, `check_env`, and the chat screen with stand-in models |

The code moves between them as a file (a zip or a git bundle) or through the project's
GitHub repository. Nothing else is shared: each laptop has its own `data/`, `models/`
and virtual environment.

## Laptop A: runs the models

**Where this section comes from.** The profile below is the `machine.md` that GitHub
Copilot wrote on that laptop on 2026-11-20. It was copied on 2026-11-23 from the
project's GitHub repository, as it stood. Nothing in it was observed from Laptop B.
Parts of it contradict each other, and contradict what is known from later:

- Its summary says Ollama is not installed. Its Ollama section says version 0.40.16 is
  installed and running. Its last line says again that Ollama is not installed.
- It lists every model as not downloaded. Yet on 2026-11-21 that laptop ingested 15
  PDFs and answered questions, which needs all of them.

Inspect that laptop again and correct this section there (`COPILOT_UPDATE.md`, step 1).

### What that laptop measured

Reported by its owner on 2026-11-23, with release 2026-11-20. Not observed from Laptop B.

- Ingesting 15 PDFs took 1,573 s, about 105 s per PDF. They gave 79 chunks and 3 figures.
- One question: the search took 81 s and the whole answer 329 s.
- `retrieval.rerank_score_threshold` was set to 0.50, as a temporary value.
- The project folder is `D:\EY Internship\docqa-code-2026-11-20\docqa`. The RBI PDFs
  are in `D:\EY Internship\docqa-docs-rbi-public-2026-11-20\docqa\docs`.
- No source file was changed there. Compared with release 2026-11-20, only
  `config.yaml` (the threshold) and `machine.md` differ.

### The profile as written on that laptop

Inspected on 2026-11-20 on this laptop. Everything below was observed by running a
command. Sections below from older inspections are legacy facts and must not be assumed
for this machine unless repeated here.

#### Re-inspected for release 2026-11-23

Inspected on 2026-11-23 by GitHub Copilot. The observations below replace the matching
values in the 2026-11-20 profile only where they were verified by commands.

- Python 3.13.3 (64-bit), pip 26.0.1.
- Ollama 0.40.17 is installed and on `PATH`. It lists `qwen2.5:7b` (4.7 GB) and
  `qwen2.5vl:7b` (6.0 GB).
- Hugging Face cache entries are present for `BAAI/bge-m3` and
  `BAAI/bge-reranker-v2-m3`.
- Docling model files are present under `models/docling` for the layout and
  layout-ONNX models, the general Docling model set and RapidOCR.
- D: has about 194 GB free (208,514,347,008 bytes of 251,604,758,528 bytes).
- The project is on branch `speed-and-new-ui` at commit
  `4b4d772b1cbc4cf45d26346399996c0c5d087a6b`.
- `python -m docqa.check_env` passed all 24 checks after stopping the old Streamlit
  processes that held Qdrant open.
- `python -m docqa.stats` reports 15 documents, 70 pages and 79 passages (64 text,
  12 tables and 3 figures). It also reports 3 earlier questions, with a median total
  time of 354 seconds.
- The 2026-11-23 `--ingested-only` refresh found 608 PDFs under the external docs
  folder, matched 15 registered documents, and left 593 never-ingested PDFs alone.
  The refresh took 404 seconds: 3 seconds reading, 384 seconds embedding and storing,
  and 17 seconds on the remaining steps. Three cached figures were skipped.
- The 2026-11-23 focused real-model tests passed all 4 selected tests in 181 seconds.
  The 8-bit benchmark took 1.2 seconds for embedding 12 texts (1.8 seconds full),
  and 1.5 seconds for reranking 12 passages (3.1 seconds full).

#### Current Summary

- Intel Core i5-1335U, 10 physical cores / 12 logical processors, 15.7 GB RAM.
- Windows 11 Home Single Language, build 26300, 64-bit.
- Python 3.13.3 (64-bit), pip 26.0.1.
- 512 GB NVMe drive; C: has about 58 GB free, D: has about 198 GB free.
- Intel Iris Xe integrated graphics only; no discrete GPU, so inference is CPU-only.
- Ollama is not installed or on `PATH`.
- Git 2.49.0.windows.3 is available.
- The external RBI PDF set is at `D:\EY Internship\docqa-docs-rbi-public-2026-11-20\docqa\docs`.

#### Operating system

- Windows 11 Home Single Language 10.0.26300, 64-bit.

#### Python

- Python 3.13.3, 64-bit.
- pip 26.0.1.
- A project virtual environment has not yet been created or verified.

#### Ollama

- Ollama 0.40.16 is installed at
  `C:\Users\dhwaj\AppData\Local\Programs\Ollama\ollama.exe` and is running.
- Its executable is not yet present in the refreshed `PATH`, but invoking it by its
  absolute path works.
- The local endpoint responds on `http://localhost:11434`.
- `ollama list` is empty: neither `qwen2.5:7b` nor `qwen2.5vl:7b` is registered.
- The user-level variables `OLLAMA_CONTEXT_LENGTH=8192` and `OLLAMA_NO_CLOUD=1`
  are set and the Ollama server was restarted.

#### Models

| Model | Status on 2026-11-20 |
|---|---|
| `qwen2.5:7b` | Not confirmed available; Ollama absent |
| `qwen2.5vl:7b` | Not confirmed available; Ollama absent |
| `BAAI/bge-m3` | Not downloaded |
| `BAAI/bge-reranker-v2-m3` | Not downloaded |
| Docling layout, table and OCR models | Not downloaded |

#### Network

- The project folder and machine can use normal package access, but this has not yet
  been fully tested for model downloads.
- No Ollama/Qwen state or model download is present until Ollama is installed and
  licensed/download approval is confirmed.

#### Behaviour of installed software found during setup

- **Python and dependency pins.** The repository's exact packages install successfully
  on Python 3.13.3. PyTorch and torchvision are CPU-only builds, and `pip check`
  reports no conflicts.
- **Qdrant local mode on Windows.** A Qdrant test collection could be created and
  deleted successfully. The project should still avoid dropping and recreating the
  collection to empty it, because deleted storage can remain open and return old points.
- **Streamlit.** Leave the project's `.streamlit/config.toml` in place; it confines
  the UI to localhost and disables usage statistics.
- **Hugging Face libraries.** Importing `docqa` switches Hugging Face to offline mode
  and disables telemetry. The separate `hf` command-line tool is not part of pipeline
  model loading.
- **Ollama.** Ollama is not installed on this machine, and the local endpoint on port
  11434 refuses connections. It must be installed and verified before steps 3 and 5
  can complete.

## Laptop B: where the code is written

Inspected on 2026-11-18, and checked again on 2026-11-19, 2026-11-20 and 2026-11-23.
The full inspection notes are in `machine.local.md` on that laptop: paths, the web
filter host by host, and the Ollama model files that were copied in. That file is kept
out of git, because it describes a company laptop and its network.

- CPU only. With 31.7 GB of RAM the planned models would fit in memory.
- Python 3.14.6 is the only interpreter. The project has its own virtual environment.
  It is kept outside the project folder, because that folder is synced to a cloud
  drive. `pip check` reports no conflicts.
- pip downloads work. Hugging Face and Ollama downloads do not: the company's web
  filter blocks both sites as a matter of policy. No mirror or other route was tried,
  and none should be.
- So the Ollama program is not installed. The files of `qwen2.5:7b` and `qwen2.5vl:7b`
  were copied in from Laptop A and checked against their checksums, but nothing there
  can load them. BGE-M3, the reranker and Docling's models are not there at all.
- `requirements.txt` and `requirements-lock.txt` were resolved on this laptop, for
  Python 3.14. Laptop A installed the same pins on Python 3.13.3.

### What was run there

| Date | What | Result |
|---|---|---|
| 2026-11-18 | `python -m docqa.check_env` | 14 passed, 5 failed: the LLM server and the four models |
| 2026-11-20 | `python -m docqa.check_env` | 14 passed, 6 failed: as before, and the new check for Docling's models |
| 2026-11-20 | `pytest -m "not slow"` | 347 passed |
| 2026-11-20 | `python -m docqa.ingest docs --limit 1 --no-figures` | Found 618 PDFs in about 30 s, then stopped: the embedding model could not be loaded |
| 2026-11-23 | `python -m docqa.check_env` | 18 passed, 6 failed: the same six. Four more packages are checked now |
| 2026-11-23 | `pytest -m "not slow"` | 460 passed |
| 2026-11-23 | The fast tests with Streamlit's packages made unimportable | All passed. Streamlit and what only it needed were then taken out of the pinned lists |
| 2026-11-23 | The text-layer check of `parsing.ocr: auto` over the 618 PDFs | About 100 ms per PDF. 7 PDFs have a scanned page; 611 can be parsed without OCR |
| 2026-11-23 | The chat screen in a browser, with stand-in models | Question, streamed answer, citations, sources with a table and a figure, refusal, document list with hover card, figures |
| 2026-11-23 | The database on its own (next section) | |

No document has been parsed, embedded or indexed on this laptop, and no real model has
run on it.

### The database on its own

Timed on 2026-11-23 with no model loaded. The search ran through the project's own code
(`docqa.retrieve`: dense search, sparse search, rank fusion) over random vectors of the
shape BGE-M3 produces: 1,024 dense numbers and about 250 word-piece weights per passage.
Median of 15 searches, `retrieval.candidates` at 30.

| Passages in the index | One search | Opening the index | Filling it |
|---|---|---|---|
| 79 (what Laptop A has today) | 6 ms | 0.03 s | 1 s |
| 3,000 | 0.2 s | 0.6 s | 42 s |
| 20,000 | 1.6 s | 4.3 s | 289 s |

Writing one question to the SQLite query log took 14 ms.

Laptop A has a different processor, so its figures will differ, but not by the factor
of ten thousand that separates 6 ms from the 81 s its search took. That time is the
embedding model and the reranker.

## Behaviour of installed software found during setup

- **Qdrant local mode on Windows** (qdrant-client 1.20.0). `delete_collection` removes
  the collection from the index but leaves its `storage.sqlite` on disk, because the
  file is still open until Python's garbage collector runs. If the file stays, a new
  collection with the same name gets the old points back. Reproduced on Laptop B, inside
  and outside the synced folder. Running `gc.collect()` and then deleting the folder
  clears it; `check_env` does this. The pipeline does not rely on dropping and
  re-creating a collection to empty it.
- **Qdrant local mode, speed.** Storing takes about 14 ms per passage, and a search
  takes time in proportion to the number of passages (table above). The client itself
  advises its server above 20,000 points. `qdrant.url` in `config.yaml` is the setting
  for that.
- **Hugging Face libraries.** `import docqa` sets `HF_HUB_OFFLINE=1` and switches
  telemetry off, confirmed in the venv. The separate `hf` command-line tool is not
  covered by that: it checks PyPI for a newer version when run. The pipeline does not
  use it.
- **8-bit weights in PyTorch 2.15.** `embedding.quantize` and `reranker.quantize` use
  `torch.ao.quantization.quantize_dynamic`. PyTorch marks it as deprecated and warns
  when it is called; it still works on a small test network. The warnings are silenced
  at that call only. It has not been tried on the real models.
- **Streamlit.** Used for the chat screen until 2026-11-23, and no longer installed.
  Left to its defaults it listened on every network interface and looked up the
  machine's public IP address through an online service; its settings file stopped
  both. The project's own chat screen listens on `127.0.0.1` and fetches nothing.

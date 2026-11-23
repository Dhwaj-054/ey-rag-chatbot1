# Machine profile

Inspected on 2026-11-20 on this laptop. Everything below was observed by running a
command. Sections below from older inspections are legacy facts and must not be assumed
for this machine unless repeated here.

## Current Summary

- Intel Core i5-1335U, 10 physical cores / 12 logical processors, 15.7 GB RAM.
- Windows 11 Home Single Language, build 26300, 64-bit.
- Python 3.13.3 (64-bit), pip 26.0.1.
- 512 GB NVMe drive; C: has about 58 GB free, D: has about 198 GB free.
- Intel Iris Xe integrated graphics only; no discrete GPU, so inference is CPU-only.
- Ollama is not installed or on `PATH`.
- Git 2.49.0.windows.3 is available.
- The external RBI PDF set is at `D:\EY Internship\docqa-docs-rbi-public-2026-11-20\docqa\docs`.

## Operating system

- Windows 11 Home Single Language 10.0.26300, 64-bit.

## Python

- Python 3.13.3, 64-bit.
- pip 26.0.1.
- A project virtual environment has not yet been created or verified.

## Ollama

- Ollama 0.40.16 is installed at
  `C:\Users\dhwaj\AppData\Local\Programs\Ollama\ollama.exe` and is running.
- Its executable is not yet present in the refreshed `PATH`, but invoking it by its
  absolute path works.
- The local endpoint responds on `http://localhost:11434`.
- `ollama list` is empty: neither `qwen2.5:7b` nor `qwen2.5vl:7b` is registered.
- The user-level variables `OLLAMA_CONTEXT_LENGTH=8192` and `OLLAMA_NO_CLOUD=1`
  are set and the Ollama server was restarted.

## Models

| Model | Status on 2026-11-20 |
|---|---|
| `qwen2.5:7b` | Not confirmed available; Ollama absent |
| `qwen2.5vl:7b` | Not confirmed available; Ollama absent |
| `BAAI/bge-m3` | Not downloaded |
| `BAAI/bge-reranker-v2-m3` | Not downloaded |
| Docling layout, table and OCR models | Not downloaded |

## Network

- The project folder and machine can use normal package access, but this has not yet
  been fully tested for model downloads.
- No Ollama/Qwen state or model download is present until Ollama is installed and
  licensed/download approval is confirmed.

## Behaviour of installed software found during setup

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

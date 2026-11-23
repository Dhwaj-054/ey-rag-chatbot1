# Instructions for GitHub Copilot

The rules for this repository are in `AGENTS.md`, in the repository root. Read it before
you change anything. It is a short version of `CLAUDE.md`, which is the authority.

The three that matter most:

1. The pipeline is local only. Do not add a call to any cloud AI service.
2. The stack is decided (Docling, Ollama with Qwen 2.5, BGE-M3, Qdrant local mode,
   SQLite, Streamlit). Ask the user before changing a component or a pinned version.
3. Do not describe something as working unless you ran it and saw it work.

`HANDOVER.md` says what has been checked and what has not. `COPILOT_PROMPT.md` is the
task of getting the build to run with the real models.

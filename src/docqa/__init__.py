"""Document Q&A prototype for BFSI documents: local, CPU-only, multimodal RAG.

Importing this package applies the project's "nothing leaves the machine" rule to the
Hugging Face libraries before any of them is loaded:

- ``HF_HUB_OFFLINE=1``: models are read from the local cache only. Nothing is fetched
  or checked online at run time, so a missing model fails loudly instead of being
  downloaded without approval.
- ``HF_HUB_DISABLE_TELEMETRY=1`` and ``DO_NOT_TRACK=1``: no usage telemetry.

``setdefault`` is used, so a value already present in the environment wins. To download
an approved model, set ``HF_HUB_OFFLINE=0`` in the shell for that one command.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

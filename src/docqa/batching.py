"""Two small helpers for running the search models (BGE-M3 and the reranker) in batches.

Both models are run from this project's own code when they are on a CPU, instead of
through the FlagEmbedding library's calls. ``docqa.retrieve.reranker`` says why and
gives the measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def runs_on_cpu(flag_model: Any) -> bool:
    """True if a FlagEmbedding model object will run on the processor and nothing else.

    On a graphics card the library's own call is kept: its trial run is how it finds a
    batch size that fits in the card's memory.
    """
    devices = getattr(flag_model, "target_devices", None)
    return isinstance(devices, (list, tuple)) and [str(device) for device in devices] == ["cpu"]


def batches_by_length(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Group positions into batches of similar length, and say in which order to run them.

    Inside a batch every input is padded to the longest one, and the padding is
    computed like real text. Sorting by length first keeps that small: with 15
    passages, batches in the order the passages arrived took a quarter longer than
    batches sorted by length.

    The batches are then put in the order of the earliest position each one holds.
    When the inputs arrive most promising first, a caller that stops early has scored
    the most promising one, and leaves out a batch whose best member came late.
    """
    by_length = sorted(range(len(lengths)), key=lambda position: (-lengths[position], position))
    batches = [by_length[start : start + batch_size] for start in range(0, len(by_length), batch_size)]
    return sorted(batches, key=min)

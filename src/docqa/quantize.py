"""Holding a model's weights as 8-bit numbers, to make it faster on a CPU.

BGE-M3 and the reranker are transformer models whose time on a CPU goes almost entirely
into multiplying large matrices of 32-bit numbers. PyTorch can keep those matrices as
8-bit integers instead ("dynamic quantization"). The model then needs about half the
memory and typically runs about twice as fast on a recent Intel or AMD processor. The
price is a small change in its outputs.

It is switched on per model with ``embedding.quantize`` and ``reranker.quantize``, and
it is off by default, because of what the change in outputs means:

- for the embedding model, vectors made with and without it do not match exactly, so
  every document is indexed again when the setting changes (the setting is part of each
  document's fingerprint);
- for the reranker, scores shift slightly, so ``retrieval.rerank_score_threshold`` has
  to be chosen again.

NOT TRIED with the real models. The conversion below is tested on a small randomly
filled model of the same kind. How much faster the real models get, and how much their
results change, has to be measured on a machine that has them.

PyTorch marks the function used here (``torch.ao.quantization.quantize_dynamic``) as
deprecated. It works in the version this project pins (2.15.0). If PyTorch is ever
upgraded, check this module first.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

log = logging.getLogger(__name__)


def _supported_engines() -> list[str]:
    import torch

    return list(torch.backends.quantized.supported_engines)


def quantize_linear_layers(module: Any) -> int:
    """Convert every linear layer inside ``module`` to 8-bit weights, in place.

    Returns how many layers were converted. Raises ``RuntimeError`` if this PyTorch
    build has no 8-bit engine for this processor.
    """
    import torch

    engines = _supported_engines()
    usable = [engine for engine in ("x86", "fbgemm", "onednn", "qnnpack") if engine in engines]
    if not usable:
        raise RuntimeError(f"This PyTorch build has no 8-bit engine (supported: {engines}).")
    torch.backends.quantized.engine = usable[0]

    before = sum(isinstance(layer, torch.nn.Linear) for layer in module.modules())
    with warnings.catch_warnings():
        # PyTorch announces on every call that this API is deprecated. That is stated
        # once, at the top of this module, and does not need repeating at each start.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        torch.ao.quantization.quantize_dynamic(
            module, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
    after = sum(isinstance(layer, torch.nn.Linear) for layer in module.modules())
    return before - after


def quantize_if_asked(model: Any, wanted: bool, name: str) -> bool:
    """Quantize ``model`` if the configuration asks for it. Returns whether it was done.

    A failure is logged and the model is used as it is: a slower answer is better than
    none.
    """
    if not wanted:
        return False
    try:
        converted = quantize_linear_layers(model)
    except Exception as exc:  # noqa: BLE001 - any failure here must leave a working model
        log.warning("%s could not be converted to 8-bit weights and runs unchanged: %s", name, exc)
        return False
    log.info("%s: %d linear layers converted to 8-bit weights", name, converted)
    return converted > 0

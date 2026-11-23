"""The 8-bit option, on a small model of the same kind as BGE-M3 and the reranker.

The real models were not available where this was written. These tests show that the
conversion works in this PyTorch build and leaves a model that still runs and still
gives nearly the same output. They cannot show how much faster the real models get.
"""

from __future__ import annotations

import logging

import pytest
import torch

from docqa.quantize import quantize_if_asked, quantize_linear_layers


def small_model() -> torch.nn.Module:
    """A randomly filled XLM-RoBERTa, the architecture both BGE models are built on."""
    from transformers import XLMRobertaConfig, XLMRobertaModel

    torch.manual_seed(0)
    config = XLMRobertaConfig(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    return XLMRobertaModel(config).eval()


def run(model: torch.nn.Module) -> torch.Tensor:
    tokens = torch.tensor([[0, 17, 42, 99, 150, 2]])
    with torch.no_grad():
        return model(input_ids=tokens).last_hidden_state


def test_linear_layers_become_8_bit_and_the_model_still_gives_nearly_the_same_output():
    model = small_model()
    before = run(model)

    converted = quantize_linear_layers(model)
    after = run(model)

    assert converted > 10
    assert not any(isinstance(layer, torch.nn.Linear) for layer in model.modules())
    assert after.shape == before.shape
    similarity = torch.nn.functional.cosine_similarity(before.flatten(), after.flatten(), dim=0)
    assert similarity > 0.98


def test_nothing_is_changed_unless_the_setting_asks_for_it():
    model = small_model()

    assert quantize_if_asked(model, False, "the model") is False
    assert any(isinstance(layer, torch.nn.Linear) for layer in model.modules())
    assert quantize_if_asked(model, True, "the model") is True
    assert not any(isinstance(layer, torch.nn.Linear) for layer in model.modules())


def test_a_failed_conversion_leaves_a_working_model_and_says_so(monkeypatch, caplog):
    model = small_model()
    before = run(model)
    monkeypatch.setattr("docqa.quantize._supported_engines", lambda: ["none"])

    with caplog.at_level(logging.WARNING):
        assert quantize_if_asked(model, True, "The reranker") is False

    assert "The reranker could not be converted to 8-bit weights and runs unchanged" in caplog.text
    assert torch.equal(run(model), before)
    with pytest.raises(RuntimeError, match="no 8-bit engine"):
        quantize_linear_layers(model)

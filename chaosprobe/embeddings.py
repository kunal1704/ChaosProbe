"""Embedding helpers for ChaosProbe.

This module provides token-level input embedding extraction for frozen
HuggingFace transformer models without generation or downstream task logic.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


SUPPORTED_MODELS = {"gpt2", "distilgpt2"}
SUPPORTED_LAYERS = {"input"}
SUPPORTED_DEVICES = {"cpu", "cuda"}


def is_huggingface_available() -> bool:
    """Return ``True`` only if both ``torch`` and ``transformers`` can import."""

    try:
        import_module("torch")
        import_module("transformers")
    except Exception:
        return False
    return True


def get_token_embeddings(
    texts: list[str],
    model_name: str = "gpt2",
    layer: str = "input",
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract token-level input embeddings for each text independently."""

    _validate_request(texts, model_name, layer, device)
    torch, AutoModel, AutoTokenizer = _load_huggingface_stack()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    torch_device = torch.device(device)
    model.to(torch_device)

    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None:
        raise ValueError(f"model '{model_name}' does not expose input embeddings")

    items = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(torch_device)

        with torch.no_grad():
            embeddings = embedding_layer(input_ids)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        items.append(
            {
                "text": text,
                "tokens": tokens,
                "embedding": embeddings.squeeze(0).detach().cpu().numpy().astype(
                    np.float64,
                    copy=False,
                ),
            }
        )

    return {
        "model_name": model_name,
        "layer": layer,
        "items": items,
    }


def _validate_request(
    texts: list[str],
    model_name: str,
    layer: str,
    device: str,
) -> None:
    if not texts:
        raise ValueError("texts must not be empty")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported model_name '{model_name}'. Supported models are: "
            f"{sorted(SUPPORTED_MODELS)}"
        )
    if layer not in SUPPORTED_LAYERS:
        raise ValueError(
            f"unsupported layer '{layer}'. Supported layers are: "
            f"{sorted(SUPPORTED_LAYERS)}"
        )
    if device not in SUPPORTED_DEVICES:
        raise ValueError("device must be either 'cpu' or 'cuda'")

    if device == "cuda":
        torch, _, _ = _load_huggingface_stack()
        if not torch.cuda.is_available():
            raise ValueError("cuda was requested but is not available")

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each text must be a non-empty, non-whitespace string")


def _load_huggingface_stack():
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "HuggingFace embedding extraction requires both torch and transformers"
        ) from exc

    return torch, transformers.AutoModel, transformers.AutoTokenizer

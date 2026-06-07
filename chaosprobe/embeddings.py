"""Frozen HuggingFace input embedding extraction for ChaosProbe.

ChaosProbe studies representation geometry rather than language generation.
This module loads supported transformer models, tokenizes supplied texts, and
returns token-level input embedding matrices. It does not request logits,
hidden states, generated text, or task-specific predictions.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


SUPPORTED_MODELS = {"gpt2", "distilgpt2"}
SUPPORTED_LAYERS = {"input"}
SUPPORTED_DEVICES = {"cpu", "cuda"}


def is_huggingface_available() -> bool:
    """Return ``True`` only when both ``torch`` and ``transformers`` import."""

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
    """Extract token-level input embeddings for each text independently.

    The returned item embeddings have shape ``T x D`` and dtype ``float64``.
    Each text is tokenized separately so prompt-level metrics can be paired back
    to the source prompt without sequence packing.
    """

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

        # Only the embedding lookup is used; no forward pass or generation.
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
    """Validate the public embedding extraction request before model loading."""

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
    """Import the optional HuggingFace stack lazily for lightweight tests."""

    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "HuggingFace embedding extraction requires both torch and transformers"
        ) from exc

    return torch, transformers.AutoModel, transformers.AutoTokenizer

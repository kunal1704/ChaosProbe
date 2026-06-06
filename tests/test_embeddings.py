import numpy as np
import pytest

from chaosprobe.embeddings import get_token_embeddings, is_huggingface_available


def test_is_huggingface_available_returns_boolean():
    assert isinstance(is_huggingface_available(), bool)


def test_empty_text_list_raises_value_error():
    with pytest.raises(ValueError):
        get_token_embeddings([])


@pytest.mark.parametrize("texts", [[""], ["   "]])
def test_empty_or_whitespace_string_raises_value_error(texts):
    with pytest.raises(ValueError):
        get_token_embeddings(texts)


def test_unsupported_model_name_raises_value_error():
    with pytest.raises(ValueError):
        get_token_embeddings(["hello"], model_name="bert-base-uncased")


def test_unsupported_layer_raises_value_error():
    with pytest.raises(ValueError):
        get_token_embeddings(["hello"], layer="hidden")


def test_unsupported_device_raises_value_error():
    with pytest.raises(ValueError):
        get_token_embeddings(["hello"], device="tpu")


def test_huggingface_integration_if_available():
    if not is_huggingface_available():
        pytest.skip("torch or transformers is not available")

    try:
        result = get_token_embeddings(["hello world"], model_name="gpt2", layer="input")
    except Exception as exc:
        pytest.skip(f"model loading unavailable in this environment: {exc}")

    assert result["model_name"] == "gpt2"
    assert result["layer"] == "input"
    assert len(result["items"]) == 1

    item = result["items"][0]
    assert item["text"] == "hello world"
    assert item["tokens"]
    assert item["embedding"].ndim == 2
    assert item["embedding"].dtype == np.float64

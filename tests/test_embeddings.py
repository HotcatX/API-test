"""Verify real pretrained embeddings and the HTTP contract, including failures."""

from math import isfinite

import pytest
import spacy
from fastapi.testclient import TestClient

from app import embedding_model
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def reference_model():
    """Load spaCy independently to verify the API returns real pretrained data."""
    return spacy.load("en_core_web_md")


@pytest.mark.parametrize("word", ["apple", "cat", "computer"])
def test_embedding_matches_pretrained_spacy_vector(word, reference_model):
    response = client.get("/embedding", params={"word": word})
    assert response.status_code == 200
    result = response.json()
    assert result["word"] == word
    assert result["model"] == "en_core_web_md"
    assert result["dimensions"] == 300
    assert len(result["embedding"]) == 300
    assert all(isinstance(value, float) and isfinite(value) for value in result["embedding"])
    assert any(value != 0 for value in result["embedding"])
    assert result["embedding"] == pytest.approx(reference_model(word)[0].vector.tolist())


def test_embedding_normalizes_whitespace_and_case():
    normalized = client.get("/embedding", params={"word": "apple"}).json()
    response = client.get("/embedding", params={"word": "  APPLE \t"})
    assert response.status_code == 200
    assert response.json() == normalized


@pytest.mark.parametrize(
    "word",
    ["", "   ", "two words", "two\twords", "two\nwords", "apple!", "123", "a-b", "a" * 101],
)
def test_embedding_rejects_invalid_words(word):
    response = client.get("/embedding", params={"word": word})
    assert response.status_code == 422


def test_embedding_requires_word():
    assert client.get("/embedding").status_code == 422


def test_embedding_rejects_word_without_vector():
    word = "qzxqzxqznotawordqzxqzxqz"
    response = client.get("/embedding", params={"word": word})
    assert response.status_code == 422
    assert "No pretrained vector" in response.json()["detail"]
    assert word in response.json()["detail"]


def test_model_is_cached_after_successful_loading():
    assert embedding_model.load_embedding_model() is embedding_model.load_embedding_model()


def test_missing_model_returns_503(monkeypatch):
    """Bypass the existing cache without clearing or mutating the real model."""
    def unavailable_model(*args, **kwargs):
        raise OSError("Simulated missing installed model")

    uncached_loader = embedding_model.load_embedding_model.__wrapped__
    monkeypatch.setattr(embedding_model.spacy, "load", unavailable_model)
    monkeypatch.setattr(embedding_model, "load_embedding_model", uncached_loader)
    response = client.get("/embedding", params={"word": "apple"})
    assert response.status_code == 503
    assert "uv sync --frozen" in response.json()["detail"]
    # An unavailable embedding model must not disable the original endpoints.
    assert client.get("/").json() == {"Hello": "World"}
    assert client.post("/generate", json={"start_word": "we", "length": 3}).status_code == 200


def test_openapi_documents_embedding():
    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/embedding"]["get"]
    assert endpoint["parameters"][0]["name"] == "word"
    assert endpoint["parameters"][0]["required"] is True
    assert "503" in endpoint["responses"]

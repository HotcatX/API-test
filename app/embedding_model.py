"""Look up pretrained spaCy word vectors without downloading models at runtime."""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

import spacy
from spacy.language import Language

MODEL_NAME = "en_core_web_md"
VECTOR_DIMENSIONS = 300
MAX_WORD_LENGTH = 100


class ModelUnavailableError(RuntimeError):
    """The installed model cannot provide the required pretrained vectors."""


@dataclass(frozen=True)
class WordEmbedding:
    """A normalized word and its unmodified pretrained vector."""

    word: str
    model: str
    dimensions: int
    embedding: list[float]


@lru_cache(maxsize=1)
def load_embedding_model() -> Language:
    """Load the installed model lazily and reuse it across subsequent requests.

    Static word vectors only need the vocabulary and tokenizer. Excluding the
    statistical pipeline components avoids running tagging, parsing, and NER.
    The model is a pinned project dependency; requests never trigger downloads.
    """
    try:
        nlp = spacy.load(
            MODEL_NAME,
            exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )
    except OSError as exc:
        raise ModelUnavailableError(
            f"The {MODEL_NAME} model is unavailable. "
            "Run 'uv sync --frozen' from the project directory or rebuild the Docker image."
        ) from exc
    if nlp.vocab.vectors_length != VECTOR_DIMENSIONS:
        raise ModelUnavailableError(
            f"The {MODEL_NAME} model must contain {VECTOR_DIMENSIONS}-dimensional vectors."
        )
    return nlp


def get_word_embedding(word: str) -> WordEmbedding:
    """Return a single alphabetic word's 300-dimensional static vector.

    Leading/trailing whitespace is removed and text is lowercased. Phrases,
    punctuation, digits, and words absent from the vector table are rejected.
    No sentence averaging, contextual embeddings, or synthetic fallback is used.
    """
    normalized = word.strip().lower()
    if not normalized or not normalized.isalpha() or len(normalized) > MAX_WORD_LENGTH:
        raise ValueError(
            f"Provide one alphabetic word of at most {MAX_WORD_LENGTH} characters "
            "without spaces, punctuation, or digits."
        )

    nlp = load_embedding_model()
    tokens = nlp.make_doc(normalized)
    if len(tokens) != 1 or not tokens[0].is_alpha:
        raise ValueError("Provide exactly one alphabetic word.")

    token = tokens[0]
    if not token.has_vector or token.vector_norm == 0:
        raise ValueError(
            f"No pretrained vector is available for '{normalized}' in {MODEL_NAME}. "
            "Try a common English word such as 'apple', 'cat', or 'computer'."
        )

    vector = token.vector.tolist()
    if len(vector) != VECTOR_DIMENSIONS or not all(isfinite(value) for value in vector):
        raise ModelUnavailableError("The model returned an invalid word vector.")
    return WordEmbedding(
        word=normalized,
        model=MODEL_NAME,
        dimensions=len(vector),
        embedding=vector,
    )

"""FastAPI endpoints for the Module 3 activity and Assignment 1 word vectors."""

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import ModelUnavailableError, get_word_embedding

app = FastAPI(
    title="Text Generation and Word Embedding API",
    description=(
        "Generate text with the Module 3 bigram model and look up pretrained "
        "300-dimensional spaCy word vectors for Assignment 1."
    ),
    version="1.0.0",
)

# Sample corpus from the Module 3 handout; unrelated to spaCy's pretrained vectors.
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(
        min_length=1,
        max_length=80,
        description="One word in the sample corpus, case-insensitive, e.g. we, the, or bigram.",
        examples=["we"],
    )
    length: int = Field(
        ge=1,
        le=200,
        strict=True,
        description="Maximum number of words including the start word; stops at a terminal word.",
        examples=[10],
    )


class TextGenerationResponse(BaseModel):
    generated_text: str


class EmbeddingResponse(BaseModel):
    word: str = Field(description="Query word after trimming whitespace and lowercasing.")
    model: str = Field(description="Name of the installed pretrained spaCy model.")
    dimensions: int = Field(description="Number of coordinates in the vector.")
    embedding: list[float] = Field(description="Full unmodified pretrained static word vector.")


@app.get("/")
def read_root() -> dict[str, str]:
    """Keep the original Module 3 Hello World endpoint."""
    return {"Hello": "World"}


@app.post("/generate", response_model=TextGenerationResponse)
def generate_text(request: TextGenerationRequest) -> TextGenerationResponse:
    """Generate bigram text; an invalid or unknown starting word returns HTTP 422."""
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TextGenerationResponse(generated_text=generated_text)


@app.get(
    "/embedding",
    response_model=EmbeddingResponse,
    responses={
        422: {"description": "Missing/invalid word, or word has no pretrained vector."},
        503: {"description": "The required spaCy vector model is unavailable."},
    },
)
def embedding(
    word: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            description=(
                "One alphabetic word (maximum 100 letters after trimming). Leading/trailing "
                "whitespace is removed and text is lowercased. Phrases, punctuation, digits, "
                "and words without pretrained vectors are rejected."
            ),
            examples=["apple"],
        ),
    ],
) -> EmbeddingResponse:
    """Return the full 300-dimensional static vector from en_core_web_md.

    This is a vocabulary lookup, so the same normalized word always returns the
    same vector. The model is installed with the project and reused after loading.
    """
    try:
        result = get_word_embedding(word)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return EmbeddingResponse(
        word=result.word,
        model=result.model,
        dimensions=result.dimensions,
        embedding=result.embedding,
    )

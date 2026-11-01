"""FastAPI endpoints for text, embeddings, classification, and GAN generation."""

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.bigram_model import BigramModel
from app.embedding_model import ModelUnavailableError, get_word_embedding
from app.digit_generator import (
    MAX_IMAGES,
    MAX_SEED,
    MODEL_NAME as GAN_MODEL_NAME,
    GeneratorUnavailableError,
    generate_digits,
)
from app.image_classifier import (
    MAX_UPLOAD_BYTES,
    ClassifierUnavailableError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageError,
    classify_image,
)

app = FastAPI(
    title="Text, Image Classification, and GAN API",
    description=(
        "Generate text with the Module 3 bigram model and look up pretrained "
        "300-dimensional spaCy word vectors for Assignment 1. "
        "Classify uploaded images with the trained Assignment 2 CIFAR-10 CNN. "
        "Generate handwritten digit images with the Assignment 3 MNIST GAN."
    ),
    version="3.0.0",
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


class ImageClassificationResponse(BaseModel):
    predicted_class: str
    class_id: int = Field(ge=0, le=9)
    confidence: float = Field(
        ge=0, le=1, description="Softmax score; not a calibrated probability of correctness."
    )
    probabilities: dict[str, float] = Field(description="Softmax scores for all ten CIFAR-10 classes.")
    model: str
    input_shape: list[int] = Field(description="Preprocessed PyTorch shape: batch, channels, height, width.")


class DigitGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=16, ge=1, le=MAX_IMAGES, strict=True)
    seed: int | None = Field(
        default=None, ge=0, le=MAX_SEED, strict=True,
        description="Noise seed for reproducibility; it does not select a digit class.",
    )


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


@app.post(
    "/classify",
    response_model=ImageClassificationResponse,
    responses={
        413: {"description": "File exceeds 5 MiB or image exceeds 16 million pixels."},
        415: {"description": "Unsupported image format."},
        422: {"description": "Missing, empty, or invalid image."},
        503: {"description": "The trained classifier is unavailable."},
    },
)
def classify(
    file: Annotated[UploadFile, File(description="A PNG, JPEG, or WebP image (at most 5 MiB).")],
) -> ImageClassificationResponse:
    """Upload one image and return its predicted class from the ten CIFAR-10 labels.

    Images are converted to RGB, resized to 64 x 64, and normalized exactly as
    during training. Arbitrary images are still assigned one of the ten classes;
    this model does not detect objects or have an unknown/reject class.
    """
    try:
        # Bound the read, even when the upload's Content-Length is absent.
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        result = classify_image(data)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClassifierUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        file.file.close()
    return ImageClassificationResponse(**result)


def digit_png_response(count: int, seed: int | None) -> Response:
    try:
        result = generate_digits(count=count, seed=seed)
    except GeneratorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=result.png,
        media_type="image/png",
        headers={
            "X-Seed": str(result.seed),
            "X-Image-Count": str(result.count),
            "X-Model": GAN_MODEL_NAME,
            "Cache-Control": "no-store",
        },
    )


GAN_RESPONSES = {
    200: {"description": "Generated grayscale PNG image.", "content": {"image/png": {}}},
    422: {"description": "Invalid count or seed."},
    503: {"description": "The trained GAN checkpoint is unavailable."},
}


@app.get("/generate-digit", response_class=Response, responses=GAN_RESPONSES)
def generate_digit(
    seed: Annotated[int | None, Query(ge=0, le=MAX_SEED, description="Optional noise seed.")] = None,
) -> Response:
    """Generate one 28 x 28 grayscale digit image with the trained GAN.

    This unconditional model cannot select a digit label. Generated images can
    be ambiguous; this endpoint does not guarantee a recognizable digit.
    """
    return digit_png_response(1, seed)


@app.post("/generate-digits", response_class=Response, responses=GAN_RESPONSES)
def generate_digit_grid(request: DigitGenerationRequest) -> Response:
    """Generate 1-64 samples in a PNG grid; each tile is 28 x 28 with 2-pixel gaps.

    The X-Seed response header records the noise seed, including when omitted
    from the request. An incomplete last grid row contains blank padding tiles.
    """
    return digit_png_response(request.count, request.seed)

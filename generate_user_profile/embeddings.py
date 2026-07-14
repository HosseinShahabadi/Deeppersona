"""Local sentence-embedding backend for DeepPersona.

This replaces the previous dependency on OpenAI's `text-embedding-ada-002`,
which is not served by OpenAI-compatible gateways such as OpenRouter.

Both the offline attribute-embedding builder (`scripts/build_embeddings.py`)
and the runtime profile-embedding path in `select_attributes.py` import from
here, so the model, pooling and normalization are guaranteed to match. If they
did not, the cosine similarities would be meaningless.

Model is configurable via DEEPPERSONA_EMBED_MODEL. Good options:
    BAAI/bge-large-en-v1.5             (default, 1024-dim)
    mixedbread-ai/mxbai-embed-large-v1 (1024-dim)
    Alibaba-NLP/gte-large-en-v1.5      (1024-dim, needs trust_remote_code=True)
    BAAI/bge-small-en-v1.5             (384-dim, much faster / lower RAM)
"""

import os
from typing import List, Optional

import numpy as np

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

MODEL_NAME = os.environ.get("DEEPPERSONA_EMBED_MODEL", DEFAULT_MODEL)

# BGE-family models are trained for asymmetric retrieval: the *query* gets an
# instruction prefix, the corpus passages do not. Here the persona summary is
# the query and the attribute paths are the corpus. Models that don't want a
# prefix (e.g. mxbai, gte) simply get an empty string.
_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
}


def _query_prefix(model_name: str) -> str:
    """Return the retrieval instruction prefix appropriate for this model."""
    override = os.environ.get("DEEPPERSONA_EMBED_QUERY_PREFIX")
    if override is not None:
        return override
    lowered = model_name.lower()
    for tag, prefix in _QUERY_PREFIXES.items():
        if tag in lowered:
            return prefix
    return ""


_model = None


def get_model():
    """Lazily load and cache the SentenceTransformer model.

    Loading is deferred so that merely importing this module (or the selector)
    does not pull a multi-hundred-MB model into memory.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for local embeddings.\n"
                "Install it with:  pip install -r requirements.txt"
            ) from exc

        kwargs = {}
        if "gte" in MODEL_NAME.lower():
            # gte-v1.5 ships custom modeling code
            kwargs["trust_remote_code"] = True

        print(f"Loading embedding model: {MODEL_NAME} (first run downloads weights)")
        _model = SentenceTransformer(MODEL_NAME, **kwargs)
    return _model


def embedding_dim() -> int:
    """Dimensionality of the configured model's output vectors."""
    return get_model().get_sentence_embedding_dimension()


def embed_passages(texts: List[str], batch_size: int = 64,
                   show_progress: bool = True) -> np.ndarray:
    """Embed corpus items (attribute paths). No instruction prefix.

    Returns an L2-normalized float32 array of shape (len(texts), dim).
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query (the persona summary), applying the model's prefix.

    Returns an L2-normalized float32 vector of shape (dim,).
    """
    model = get_model()
    vector = model.encode(
        _query_prefix(MODEL_NAME) + text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype=np.float32)

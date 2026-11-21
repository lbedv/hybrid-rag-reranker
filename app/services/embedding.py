import hashlib
import math
from typing import Sequence
from openai import OpenAI

from app.core.config import settings


class EmbeddingService:
    def __init__(self, api_key: str | None = None, model: str | None = None, dim: int | None = None):
        raw_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.api_key = raw_key.strip() if (raw_key and raw_key.strip()) else None
        self.model = model or settings.EMBEDDING_MODEL
        self.dim = dim or settings.EMBEDDING_DIM
        self._client: OpenAI | None = None
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def is_live(self) -> bool:
        """True if a real OpenAI client is configured."""
        return self._client is not None

    def _generate_deterministic_embedding(self, text: str, dim: int | None = None) -> list[float]:
        """
        Generate a deterministic unit-length pseudo-embedding vector from text hash.
        Used for offline mode, CI, and deterministic test runs without OpenAI API costs.
        """
        dim = dim or self.dim
        # Seed generator with sha256 hash
        hasher = hashlib.sha256(text.encode("utf-8"))
        seed = int.from_bytes(hasher.digest()[:8], byteorder="big")
        
        # Linear congruential generator for reproducible pseudo-random floats
        values: list[float] = []
        state = seed
        for _ in range(dim):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            # Map uint64 to [-1.0, 1.0]
            val = ((state >> 32) / 2147483648.0) - 1.0
            values.append(val)

        # L2 normalize to exact unit length
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def get_embedding(self, text: str) -> list[float]:
        """Compute embedding for a single text."""
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: Sequence[str], batch_size: int = 50) -> list[list[float]]:
        """Compute embeddings for a sequence of texts, batching requests."""
        if not texts:
            return []

        # If live OpenAI client is available, use API
        if self._client is not None:
            results: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                response = self._client.embeddings.create(input=batch, model=self.model)
                batch_embeddings = [item.embedding for item in response.data]
                results.extend(batch_embeddings)
            return results

        # Offline / deterministic fallback
        return [self._generate_deterministic_embedding(t) for t in texts]


embedding_service = EmbeddingService()

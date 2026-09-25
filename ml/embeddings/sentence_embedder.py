import os
import hashlib
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    """
    Sentence embedding engine leveraging transformer representations.
    Features:
    - Lazy model loading
    - In-memory and disk caching of computed vectors
    - Fast batch processing
    - Cosine similarity calculations
    """

    _instance: Optional["SentenceEmbedder"] = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = "data/cache/embeddings",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Optional[SentenceTransformer] = None
        self._memory_cache: dict = {}

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode text or list of texts into dense semantic vectors.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        if not text_list:
            return np.empty((0, 384), dtype=np.float32)

        indices_to_compute = []
        texts_to_compute = []
        results = [None] * len(text_list)

        for i, text in enumerate(text_list):
            if not text or not text.strip():
                results[i] = np.zeros(384, dtype=np.float32)
                continue

            cache_key = self._get_cache_key(text)
            if use_cache and cache_key in self._memory_cache:
                results[i] = self._memory_cache[cache_key]
                continue

            # Check disk cache
            if use_cache and self.cache_dir:
                disk_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                if os.path.exists(disk_path):
                    try:
                        vec = np.load(disk_path)
                        self._memory_cache[cache_key] = vec
                        results[i] = vec
                        continue
                    except Exception:
                        pass

            indices_to_compute.append(i)
            texts_to_compute.append(text)

        if texts_to_compute:
            computed_vectors = self.model.encode(
                texts_to_compute,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for idx, vec in zip(indices_to_compute, computed_vectors):
                results[idx] = vec
                cache_key = self._get_cache_key(text_list[idx])
                self._memory_cache[cache_key] = vec
                if use_cache and self.cache_dir:
                    disk_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                    try:
                        np.save(disk_path, vec)
                    except Exception:
                        pass

        output_matrix = np.array(results, dtype=np.float32)
        return output_matrix[0] if is_single else output_matrix

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between 1D or 2D normalized vector arrays.
        """
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.dot(a, b.T)

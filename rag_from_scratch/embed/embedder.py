import ollama
import numpy as np


class OllamaEmbedder:
    """调用本地 Ollama embedding 模型，将文本转为向量"""

    def __init__(self, model: str = "qwen3-embedding"):
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        批量向量化文本。

        Args:
            texts: 文本列表

        Returns:
            shape (n, dim) 的 float32 numpy 数组
        """
        resp = ollama.embed(model=self.model, input=texts)
        return np.array(resp.embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """单条文本向量化，返回 shape (dim,) 的向量"""
        return self.embed([text])[0]

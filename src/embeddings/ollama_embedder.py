from langchain_ollama import OllamaEmbeddings

from src.config import get_settings
from src.embeddings.base_embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def initialize_embedding_model(self) -> None:
        settings = get_settings()
        kwargs = {
            "model": settings.embedding_model,
            "base_url": settings.llm_provider_base_url,
            "dimensions": settings.embedding_dim,
        }
        self.model = OllamaEmbeddings(**kwargs)

from langchain_openai import OpenAIEmbeddings

from src.config import get_settings
from src.embeddings.base_embedder import BaseEmbedder


class GitHubEmbedder(BaseEmbedder):
    """
    Embedding provider backed by Github
    """

    def initialize_embedding_model(self) -> None:
        settings = get_settings()
        kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.token,
            "base_url": settings.llm_provider_base_url,
        }
        self.model = OpenAIEmbeddings(**kwargs)

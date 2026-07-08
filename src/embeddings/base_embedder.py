from abc import ABC, abstractmethod

import structlog
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__file__)


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding providers.
    """

    model: Embeddings

    @abstractmethod
    def initialize_embedding_model(self) -> None:
        """
        Abstract method to initialize embedding model
        """
        ...

    async def embed_query(self, query: str) -> list[float]:
        """
        Encode a single text string.

        Args:
            text: The text to encode.

        Returns:
            A single float vector.
        """

        try:
            if not query:
                raise ValueError("Query must not be empty")

            results = await self.model.aembed_query(query)
            logger.debug("query.vector_embedded_successfully", query=query[:100])
            return results
        except Exception as e:
            logger.exception(
                "query.embedding_creation_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.

        Runs the CPU/GPU-bound encoding in a thread-pool executor to avoid
        blocking the asyncio event loop.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            A list of float vectors, one per input text.

        Raises:
            ValueError: If ``texts`` is empty.
        """

        try:
            if not texts:
                raise ValueError("texts must not be an empty list.")

            vectors = await self.model.aembed_documents(texts)
            logger.debug("documents.embedding_created_successfully")
            return vectors
        except Exception as e:
            logger.exception(
                "documents.embedding_creation_failed",
                error=str(e),
            )
            raise

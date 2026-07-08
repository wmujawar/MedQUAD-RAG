import asyncio
import uuid

import structlog
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from langgraph.store.base import Embeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from src.config import get_settings

logger = structlog.get_logger(__file__)


class QDrantStore:
    def __init__(self, embedding_model: Embeddings) -> None:
        settings = get_settings()

        self._embedding_model = embedding_model
        self._embedding_dim = settings.embedding_dim
        self._collection_name = settings.qdrant_collection
        self._client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_key, timeout=120
        )

    @staticmethod
    def _stable_uuid(text: str) -> str:
        """Generate a deterministic UUID from document content."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def get_vector_store(self) -> VectorStore:

        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim, distance=Distance.COSINE
                ),
            )

        return QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._embedding_model,
        )

    async def add_documents(
        self, docs: list[Document], vectors: list[list[float]]
    ) -> None:

        if len(docs) != len(vectors):
            logger.warning(
                "add_documents.failed",
                reason="doc vector mismatch",
            )
            raise ValueError(
                f"Mismatch: {len(docs)} docs and {len(vectors)} vectors provided."
            )

        try:
            points = [
                models.PointStruct(
                    id=self._stable_uuid(doc.page_content),
                    vector=v,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for doc, v in zip(docs, vectors)
            ]

            logger.debug("add_documents.upserting", point_count=len(points))

            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.upsert(
                    collection_name=self._collection_name, points=points
                ),
            )

            logger.info("add_documents.success", points_upserted=len(points))

        except Exception as e:
            logger.exception("add_documents.failed", error=str(e), exc_info=True)
            raise

    def search(self, *, query: str, k: int = 3, fetch_k: int = 10) -> list[Document]:
        """
        Fetch document using Maximum Marginal Relevance (MMR) retrieval.

        Args:
            query (str): The search query text.
            k (int): Number of diverse documents to return.
            fetch_k (int): Total number of candidate documents to initially fetch for evaluation.

        Returns:
            List[Document]: A list of retrieved, distinct documents.
        """

        if not query.strip():
            logger.warning("search.failed", reason="empty_query")
            raise ValueError("Search query must not be empty.")

        try:
            retriever = self.get_vector_store().as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k}
            )

            docs = retriever.invoke(query)

            logger.debug(
                "search.success",
                docs_returned=len(docs),
            )

            return docs

        except Exception as e:
            logger.exception("search.failed", error=str(e))
            raise

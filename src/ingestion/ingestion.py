from pathlib import Path
from typing import Iterator

import structlog
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.cache import RedisStore
from src.vector_store.qdrant_store import QDrantStore

DATA_DIR = Path(__file__).parent.parent.parent / "data"

logger = structlog.get_logger(__file__)


class Ingest:
    def __init__(
        self, embedder: BaseEmbedder, vector_store: QDrantStore, redis: RedisStore
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._redis = redis

    def _load_xml_documents(self) -> Iterator[Document]:
        loader = DirectoryLoader(
            path=DATA_DIR.as_posix(),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False,
        )

        docs: Iterator[Document] = iter([])

        try:
            docs = loader.lazy_load()
            logger.debug("xml_documents.ingested")

        except Exception as e:
            logger.exception("xml_documents.load_failed", error=str(e))
            raise

        return docs

    def _chunk_text(self, documents: Iterator[Document]) -> list[Document]:
        settings = get_settings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            path: Path = Path(chunk.metadata.get("source", ""))

            chunk.metadata["folder"] = path.parent.name
            chunk.metadata["filename"] = path.name

        logger.debug("chunks.created", chunk_count=len(chunks))

        return chunks

    async def ingest_document(self) -> None:
        documents = self._load_xml_documents()

        chunks = self._chunk_text(documents)

        batch = await self._redis.get_batch(chunks)

        documents_to_embed = [doc for doc, vector in batch if vector is None]

        if not documents_to_embed:
            return

        # embed the document and get vector
        texts = [text.page_content for text in documents_to_embed]
        vectors = await self._embedder.embed_documents(texts)

        # add documents to the vector store
        await self._vector_store.add_documents(documents_to_embed, vectors)

        # Store the vector's in Redis
        await self._redis.set_batch(data=list(zip(documents_to_embed, vectors)))

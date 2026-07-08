import hashlib
import json
from typing import Optional

import structlog
from langchain_core.documents import Document
from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.client import Pipeline

from src.config import get_settings

logger = structlog.get_logger(__file__)


def _sha256(text: str) -> str:
    """
    Return the SHA-256 hex digest of *text* encoded as UTF-8.

    Args:
        text: The string to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RedisStore:
    """Async Redis cache for document chunk embeddings.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379``).
        ttl: Time-to-live for cached entries in seconds.
        key_prefix: Namespace prefix applied to every Redis key.
    """

    def __init__(
        self, *, redis_url: str, ttl: Optional[int] = None, key_prefix: str = "emb:"
    ) -> None:
        pass
        self._ttl = ttl
        self._prefix = key_prefix

        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=20,
            decode_responses=True,
        )
        self._redis: Redis = Redis(connection_pool=self._pool)

    def _generate_key(self, text: str) -> str:
        """
        Build the Redis key for a given chunk text.

        Args:
            text: Raw chunk text.

        Returns:
            Prefixed SHA-256 key string.
        """
        return f"{self._prefix}{_sha256(text=text)}"

    async def get(self, doc: Document) -> list[float] | None:
        """
        Retrieve a cached embedding vector for *document*.

        Args:
            doc: Document chunk.

        Returns:
            The cached embedding vector, or ``None`` on a cache miss.
        """
        key = self._generate_key(doc.page_content)

        try:
            raw = await self._redis.get(key)

            if raw is None:
                logger.debug("document.vector_embedding_not_found", doc_id=doc.id)
                return None
            else:
                logger.debug("document.vector_embedding_found", doc_id=doc.id)
                return json.loads(raw)
        except Exception as e:
            logger.exception(
                "vector_embeddings.fetch_failed",
                error=str(e),
            )
            raise

    async def set(self, doc: Document, vector: list[float]) -> None:
        """
        Store an embedding vector for *text*.

        Args:
            doc: Document chunk.
            vector: The embedding float vector.

        """
        key = self._generate_key(doc.page_content)

        try:
            await self._redis.set(key, json.dumps(vector), ex=self._ttl)
            logger.debug("document.vector_embedding_stored", doc_id=doc.id)
        except Exception as e:
            logger.exception(
                "vector_embedding.store_failed",
                error=str(e),
            )
            raise

    async def get_batch(
        self, docs: list[Document]
    ) -> list[tuple[Document, list[float] | None]]:
        """
        Retrieve embeddings for a batch of texts in a single pipeline call.

        Uses a Redis pipeline to send all GET commands in one round-trip.

        Args:
            docs: List of chunk.

        Returns:
            Mapping from Document to its cached vector (or ``None`` if not cached).
        """

        if not docs:
            logger.debug("cache.no_documents")
            return []

        keys = [self._generate_key(doc.page_content) for doc in docs]

        try:
            pipeline: Pipeline = self._redis.pipeline()

            for key in keys:
                pipeline.get(key)

            raw_values: list[str | None] = await pipeline.execute()

            items: list[tuple[Document, list[float] | None]] = [
                (doc, None if vector is None else json.loads(vector))
                for doc, vector in zip(docs, raw_values, strict=True)
            ]

            logger.debug(
                "document.embeddings_fetched",
                total=len(docs),
                hits=sum(v is not None for _, v in items),
                misses=sum(v is None for _, v in items),
            )

            return items

        except Exception as e:
            logger.exception(
                "document.embeddings_fetch_failed",
                doc_count=len(docs),
                error=str(e),
            )
            raise

    async def set_batch(self, data: list[tuple[Document, list[float]]]) -> None:
        """
        Store multiple embeddings in a single Redis pipeline call.

        Args:
            data: Iterable of (document, vector) tuples to cache.
        """
        if not data:
            return

        try:
            pipeline: Pipeline = self._redis.pipeline()

            for doc, vector in data:
                key = self._generate_key(doc.page_content)
                pipeline.set(key, json.dumps(vector), ex=self._ttl)

            await pipeline.execute()
            logger.debug(
                "document.vectors_cached",
                doc_count=len(data),
                ttl=self._ttl,
            )
        except Exception as e:
            logger.exception(
                "document.vectors_cache_failed",
                doc_ids=[doc.id for doc, _ in data],
                doc_count=len(data),
                error=str(e),
            )
            raise

    async def ping(self) -> bool:
        """
        Check Redis connectivity.

        Returns:
            ``True`` if Redis responds to PING, ``False`` otherwise.
        """

        try:
            self._redis.ping()
            logger.debug("connection_pool.ping_acknowledged")
            return True
        except Exception:
            logger.error(
                "redis.connection_failed", reason="timeout_or_connection_refused"
            )
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        logger.debug("cache_store.closing_connection_pool")
        await self._redis.aclose()
        await self._pool.aclose()
        logger.debug("connection.closed")


_redis_store: RedisStore | None = None


def initiate_redis_store() -> RedisStore:
    """Create (or return the cached) EmbeddingCache singleton.

    Called once during FastAPI lifespan startup.

    Returns:
        RedisStore: The singleton cache instance.
    """

    global _redis_store

    if _redis_store is None:
        settings = get_settings()
        _redis_store = RedisStore(
            redis_url=settings.redis_url, ttl=settings.embedding_cache_ttl
        )

    return _redis_store


def get_redis_store() -> RedisStore:
    """Return the already-initialised cache singleton.

    Returns:
        RedisStore: The singleton cache instance.

    Raises:
        RuntimeError: If `initiate_redis_store()` was not called at startup.
    """
    if _redis_store is None:
        raise RuntimeError(
            "RedisStore has not been initialised. "
            "Call initiate_redis_store() during application startup."
        )

    return _redis_store

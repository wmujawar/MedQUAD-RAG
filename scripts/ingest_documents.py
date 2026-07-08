import asyncio

from src.config import get_settings
from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.cache import get_redis_store, initiate_redis_store
from src.embeddings.github_embedder import GitHubEmbedder
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.ingestion.ingestion import Ingest
from src.utils.container import Container
from src.utils.logger import setup_logging
from src.vector_store.qdrant_store import QDrantStore

setup_logging()
settings = get_settings()
initiate_redis_store()


embedder_container = Container()
embedder_container.register("ollama", OllamaEmbedder, True)
embedder_container.register("github_openai", GitHubEmbedder, True)


async def main():
    embedder: BaseEmbedder = embedder_container.resolve(settings.embedding_provider)
    embedder.initialize_embedding_model()

    vector_store = QDrantStore(embedder.model)
    vector_store.get_vector_store()

    redis = get_redis_store()

    ingest = Ingest(embedder, vector_store, redis)
    await ingest.ingest_document()


if __name__ == "__main__":
    asyncio.run(main())

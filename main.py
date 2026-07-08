from langfuse import Langfuse, get_client

from src.config import get_settings
from src.embeddings.github_embedder import GitHubEmbedder
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.llm.github_llm import GithubModelProvider
from src.llm.ollama_llm import OllamaModelProvider
from src.rag.pipeline import Pipeline, fetch_answer
from src.utils.container import Container
from src.utils.logger import setup_logging
from src.vector_store.qdrant_store import QDrantStore

embedder_container = Container()
embedder_container.register("ollama", OllamaEmbedder, True)
embedder_container.register("github_openai", GitHubEmbedder, True)

llm_container = Container()
llm_container.register("ollama", OllamaModelProvider, True)
llm_container.register("github_openai", GithubModelProvider, True)


setup_logging()
settings = get_settings()

Langfuse(
    host=settings.langfuse_host,
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value()
    if settings.langfuse_secret_key is not None
    else None,
)

langfuse = get_client()


def main():

    model_provider = llm_container.resolve(settings.generation_provider)
    model = model_provider.get_model()

    embedder = embedder_container.resolve(settings.embedding_provider)
    embedder.initialize_embedding_model()

    qdrant_store = QDrantStore(embedding_model=embedder.model)
    qdrant_store.get_vector_store()

    graph = Pipeline(vector_store=qdrant_store, llm=model).build_graph()

    while True:
        question: str = input("Enter your question: ")

        if question == "exit":
            break

        answer = fetch_answer(graph, question, 1)

        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

import structlog
from fastapi import APIRouter, Body, status
from fastapi.responses import JSONResponse
from openai import APIStatusError

from src.config import get_settings
from src.embeddings.github_embedder import GitHubEmbedder
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.llm.github_llm import GithubModelProvider
from src.llm.ollama_llm import OllamaModelProvider
from src.rag.pipeline import Pipeline, fetch_answer
from src.utils.container import Container
from src.vector_store.qdrant_store import QDrantStore

router = APIRouter()

embedder_container = Container()
embedder_container.register("ollama", OllamaEmbedder, True)
embedder_container.register("github_openai", GitHubEmbedder, True)

llm_container = Container()
llm_container.register("ollama", OllamaModelProvider, True)
llm_container.register("github_openai", GithubModelProvider, True)

logger = structlog.get_logger(__file__)

settings = get_settings()


@router.post("/ask")
def get_answer(question: str = Body(..., embed=True)):
    model_provider = llm_container.resolve(settings.generation_provider)
    model = model_provider.get_model()

    embedder = embedder_container.resolve(settings.embedding_provider)
    embedder.initialize_embedding_model()

    qdrant_store = QDrantStore(embedding_model=embedder.model)

    try:
        qdrant_store.get_vector_store()
    except APIStatusError as e:
        logger.exception("openai.api_key.invalid", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"message": "Invalid LLM Provider API key"},
        )

    graph = Pipeline(vector_store=qdrant_store, llm=model).build_graph()

    try:
        return fetch_answer(graph, question, 1)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"message": str(e)}
        )

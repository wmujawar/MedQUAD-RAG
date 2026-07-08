import structlog
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import get_settings
from src.llm.base_llm import BaseModelProvider

logger = structlog.get_logger(__file__)


class OllamaModelProvider(BaseModelProvider):
    """
    LLM provider backed by Azure OpenAI.
    """

    def get_model(self) -> BaseChatModel:
        super().__init__()
        settings = get_settings()
        self._model_name = settings.generator_model
        self._base_url = settings.llm_provider_base_url

        try:
            return ChatOllama(
                base_url=self._base_url, model=self._model_name, temperature=0.7
            )
        except Exception as e:
            logger.exception(
                "openai_model.initialization_failed",
                error=str(e),
            )
            raise

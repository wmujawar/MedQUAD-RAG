import structlog
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.llm.base_llm import BaseModelProvider

logger = structlog.get_logger(__file__)


class GithubModelProvider(BaseModelProvider):
    """
    LLM provider backed by a locally running Ollama instance.
    """

    def get_model(self) -> BaseChatModel:
        settings = get_settings()
        self._model_name = settings.generator_model
        self._base_url = settings.llm_provider_base_url
        self._token = settings.token

        try:
            return ChatOpenAI(
                base_url=self._base_url,
                model=self._model_name,
                temperature=0.7,
                api_key=self._token,
            )
        except Exception as e:
            logger.exception(
                "openai_model.initialization_failed",
                error=str(e),
            )
            raise

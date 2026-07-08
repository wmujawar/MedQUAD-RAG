from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class BaseModelProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement the `model` property.
    """

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """
        Returns a LangChain-compatible chat model instance.
        All providers must return a BaseChatModel so they are
        interchangeable across the codebase.
        """
        ...

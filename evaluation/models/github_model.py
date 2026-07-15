from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class GithubModel(DeepEvalBaseLLM):
    def __init__(
        self, model: str, token: SecretStr | None = None, temperature: int = 0, **kwargs
    ):
        self.model_name = model
        self.temperature = temperature
        self.token = token

        if not self.token:
            raise ValueError(
                "Github token is required. either pass the token or set the GITHUB_TOKEN environment variable."
            )

        self.model = ChatOpenAI(
            model=self.model_name,
            api_key=self.token,
            base_url="https://models.inference.ai.azure.com",
            temperature=self.temperature,
            **kwargs,
        )

    def load_model(self):
        return self

    def generate(self, prompt: str):
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt):
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return f"GitHub Models ({self.model_name})"

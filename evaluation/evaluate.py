import json
from functools import partial
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from evaluation.models.github_model import GithubModel
from src.config import get_settings
from src.embeddings.github_embedder import GitHubEmbedder
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.llm.github_llm import GithubModelProvider
from src.llm.ollama_llm import OllamaModelProvider
from src.rag.pipeline import Pipeline, RAGState
from src.utils.container import Container
from src.vector_store.qdrant_store import QDrantStore

GOLDEN_FILE = Path(__file__).parent / "goldens.json"
METRICS_THRESHOLD = 0.7

settings = get_settings()

kwargs = {
    "model": settings.generator_model,
    "temperature": 0.0,
}

model_container = Container()
model_container.register("ollama", partial(OllamaModel, **kwargs), True)
model_container.register(
    "github_openai", partial(GithubModel, token=settings.token, **kwargs), True
)

embedder_container = Container()
embedder_container.register("ollama", OllamaEmbedder, True)
embedder_container.register("github_openai", GitHubEmbedder, True)

llm_container = Container()
llm_container.register("ollama", OllamaModelProvider, True)
llm_container.register("github_openai", GithubModelProvider, True)

model = model_container.resolve(settings.generation_provider)


def load_golders():
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def ask_question(question):

    initial_state: RAGState = {
        "question": question,
        "rewritten_attempts": 0,
        "answer": "",
        "input_guard_error": "",
        "output_guard_error": "",
        "retrieved_docs": [],
        "is_grounded": None,
        "is_relevant": None,
        "rewritten_question": "",
    }

    model_provider = llm_container.resolve(settings.generation_provider)
    model = model_provider.get_model()

    embedder = embedder_container.resolve(settings.embedding_provider)
    embedder.initialize_embedding_model()

    qdrant_store = QDrantStore(embedding_model=embedder.model)
    qdrant_store.get_vector_store()

    graph = Pipeline(vector_store=qdrant_store, llm=model).build_graph()
    config = {"configurable": {"thread_id": 1}}

    final_state = graph.invoke(initial_state, config=config)

    answer = final_state.get("answer", "")
    retrieved_context = [
        doc.page_content for doc in final_state.get("retrieved_docs", [])
    ]

    return answer, retrieved_context


def main():

    metrics = [
        ContextualPrecisionMetric(threshold=METRICS_THRESHOLD, model=model),
        ContextualRecallMetric(threshold=METRICS_THRESHOLD, model=model),
        ContextualRelevancyMetric(threshold=METRICS_THRESHOLD, model=model),
        AnswerRelevancyMetric(threshold=METRICS_THRESHOLD, model=model),
        FaithfulnessMetric(threshold=METRICS_THRESHOLD, model=model),
    ]

    # load golders
    pairs = load_golders()

    test_case: list[LLMTestCase] = []
    # Iterate through goldens
    for pair in pairs:
        answer, context = ask_question(pair["input"])
        test_case.append(
            LLMTestCase(
                input=pair["input"],
                expected_output=pair["expected_output"],
                actual_output=answer,
                retrieval_context=context,
            )
        )

    # Evaluate
    results = evaluate(
        test_case,
        metrics,
        async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
    )

    # Save result to json file
    summary = []

    for result in results.test_results:
        summary.append(
            {
                "question": result.input,
                "expected_output": result.expected_output,
                "actual_output": result.actual_output,
                "success": result.success,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "passed": m.success,
                        "reason": m.reason,
                    }
                    for m in result.metrics_data
                ],
            }
        )
    results_path = Path("eval_results.json")
    results_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nResults saved to {results_path}.")


if __name__ == "__main__":
    main()

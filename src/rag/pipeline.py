import sqlite3
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import get_settings
from src.guardrails.guardrails import GuardRailChecker, GuardResult
from src.prompts.prompt_templates import (
    HALLUCINATION_CHECK_PROMPT,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_CHECK_PROMPT,
    RESPONSE_GENERATION_PROMPT,
)
from src.rag.models import GradeDocuments, HallucinationCheck
from src.vector_store.qdrant_store import QDrantStore

MAX_REWRITE_QUERY_ATTEMPT = 3


class RAGState(TypedDict):
    question: str
    retrieved_docs: list[Document]
    is_relevant: bool | None
    rewritten_question: str
    rewritten_attempts: int
    answer: str | None
    is_grounded: bool | None
    input_guard_error: str
    output_guard_error: str


class Pipeline:
    def __init__(
        self,
        *,
        vector_store: QDrantStore,
        llm: BaseChatModel,
        sqllite_checkpoint: str = "checkpoint.db",
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._checkpoint = sqllite_checkpoint
        self._guard = GuardRailChecker()

    def _is_retrieval_attempt_exhausted(self, state: RAGState):
        attempts = state.get("rewritten_attempts", 0)

        return not attempts >= MAX_REWRITE_QUERY_ATTEMPT

    def _retrieve(self, state: RAGState) -> dict:
        """Retrieves relevant documents from the vector store based on the user's query.

        This node acts as the entry point of the RAG pipeline. It extracts the current
        query from the state, queries ChromaDB, and updates the state with the fetched
        documents.

        Args:
            state (RAGState): The current state of the graph, containing the key 'question'.

        Returns:
            dict: A state update dictionary containing the documents with the
                  list of retrieved context strings or LangChain Document objects.
        """
        question = state["rewritten_question"] or state["question"]

        result: GuardResult = self._guard.validate_input(question)

        if result.passed:
            response = self._vector_store.search(query=result.text)
            return {"retrieved_docs": response}
        else:
            return {"input_guard_error": result.error}

    def _grade_documents(self, state: RAGState) -> dict:
        """Evaluates the relevancy of the retrieved documents to the user's query.

        Acts as a routing node (Document Grader) to filter out noise. If the documents
        are relevant, the graph routes toward generation. If they are deemed irrelevant,
        it flags the state to route to the END.

        Args:
            state (RAGState): The current state of the graph, containing 'question'
                                and 'documents'.

        Returns:
            dict: A state update dictionary containing a 'is_relevant' flag
                  to guide conditional routing.
        """
        question = state["question"]
        documents = state["retrieved_docs"]

        llm = self._llm.with_structured_output(GradeDocuments)
        relevancy_check_prompt = ChatPromptTemplate.from_messages(
            [("system", RELEVANCE_CHECK_PROMPT)]
        )

        chain = relevancy_check_prompt | llm

        response = chain.invoke({"question": question, "document": documents})

        return {"is_relevant": response.is_relevant}

    def _query_rewrite(self, state: RAGState) -> dict:
        """Optimizes and reformulates the user query to improve retrieval performance.

        This node is triggered when the previous retrieval failed to yield relevant context.
        It leverages an LLM to rewrite the raw question into a more descriptive or
        search-optimized query before looping back to the retrieval node.

        Args:
            state (RAGState): The current state of the graph, containing the original
                                'question' and execution metadata like 'loop_count'.

        Returns:
            dict: A state update dictionary containing the newly optimized 'question'
                    and an incremented 'loop_count' to prevent infinite execution cycles.
        """

        attempts = state.get("rewritten_attempts", 0)
        prompt = ChatPromptTemplate.from_messages([("system", QUERY_REWRITE_PROMPT)])

        chain = prompt | self._llm | StrOutputParser()

        response = chain.invoke({"question": state["question"]})

        return {"rewritten_attempts": attempts + 1, "rewritten_question": response}

    def _generate_answer(self, state: RAGState) -> dict:
        """Synthesizes a response using the retrieved documents as contextual grounding.

        This node passes both the query and the validated relevant documents to an LLM,
        instructing it to formulate a precise answer based strictly on the provided context.

        Args:
            state (RAGState): The current state of the graph, containing 'question'
                                and the validated 'documents'.

        Returns:
            dict: A state update dictionary containing the key 'response' with the
                    generated text answer.
        """
        question = state["rewritten_question"] or state["question"]
        documents = state["retrieved_docs"]

        prompt = ChatPromptTemplate.from_messages(
            [("system", RESPONSE_GENERATION_PROMPT)]
        )

        chain = prompt | self._llm | StrOutputParser()

        response = chain.invoke({"question": question, "context": documents})

        result: GuardResult = self._guard.validate_output(response)

        if result.passed:
            return {"answer": response}
        else:
            return {"output_guard_error": result.error}

    def _hallucination_check(self, state: RAGState) -> dict:
        """Validates the generated response against the context to detect hallucinations.

        Acts as the final guardrail (Hallucination Checker) in the Self-RAG architecture.
        It uses an evaluation LLM to determine if the generated answer is fully grounded
        in and supported by the retrieved documents.

        Args:
            state (RAGState): The current state of the graph, containing 'documents'
                                and the generated 'response'.

        Returns:
            dict: A state update dictionary containing a 'is_grounded' flag
                    used to route the graph to END or back to retry.
        """

        context = state["retrieved_docs"]
        answer = state["answer"]

        llm = self._llm.with_structured_output(HallucinationCheck)

        prompt = ChatPromptTemplate.from_messages(
            [("system", HALLUCINATION_CHECK_PROMPT)]
        )

        chain = prompt | llm

        response = chain.invoke({"context": context, "response": answer})

        return {"is_grounded": response.is_grounded}

    def _route_no_relevant(self, state: RAGState):
        is_relevant = state["is_relevant"]

        if is_relevant:
            return "generate_answer"
        else:
            return "end"

    def _route_after_generation(self, state: RAGState) -> str:
        no_answer_response = "I am sorry, but the provided context does not contain enough information to answer this question."
        if state["answer"] == no_answer_response:
            return "end"
        else:
            return "hallucination_check"

    def _route_hallucination_check(self, state: RAGState) -> str:
        attempts = state["rewritten_attempts"]

        if state["answer"] and state["is_grounded"]:
            return "end"

        if attempts >= MAX_REWRITE_QUERY_ATTEMPT:
            return "end"
        else:
            return "rewrite_query"

    def _route_attempt_check(self, state: RAGState):
        attempts = state.get("rewritten_attempts", 0)

        if attempts >= MAX_REWRITE_QUERY_ATTEMPT:
            return "end"
        else:
            return "retrieve"

    def build_graph(self):
        conn: sqlite3.Connection = sqlite3.connect(
            self._checkpoint, check_same_thread=False
        )
        checkpointer: SqliteSaver = SqliteSaver(conn)

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_retrieval", self._grade_documents)
        graph.add_node("rewrite_query", self._query_rewrite)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("grade_hallucination", self._hallucination_check)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_retrieval")

        graph.add_conditional_edges(
            "grade_retrieval",
            self._route_no_relevant,
            {"generate_answer": "generate_answer", "end": END},
        )

        graph.add_conditional_edges(
            "generate_answer",
            self._route_after_generation,
            {"hallucination_check": "grade_hallucination", "end": END},
        )

        graph.add_conditional_edges(
            "grade_hallucination",
            self._route_hallucination_check,
            {"rewrite_query": "rewrite_query", "end": END},
        )

        graph.add_conditional_edges(
            "rewrite_query",
            self._route_attempt_check,
            {"retrieve": "retrieve", "end": END},
        )

        return graph.compile(checkpointer=checkpointer)


settings = get_settings()

Langfuse(
    host=settings.langfuse_host,
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value()
    if settings.langfuse_secret_key is not None
    else None,
)

langfuse = get_client()


def fetch_answer(graph: CompiledStateGraph, question: str, thread_id: int):
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
    langfuse_handler = CallbackHandler()

    response = graph.invoke(
        initial_state,
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
            "run_name": "MedQUAD RAG",
        },
    )

    if response.get("input_guard_error"):
        raise Exception("Question has gibbish text or toxic language.")

    if not response.get("is_relevant"):
        raise Exception("Retrieved context is not relavent.")

    if response.get("output_guard_error"):
        raise Exception("Response has toxic language.")

    if not response.get("is_grounded"):
        raise Exception("Unable to answer the question with confidence.")

    return response.get("answer")

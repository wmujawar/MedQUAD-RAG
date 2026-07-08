RELEVANCE_CHECK_PROMPT = """
You are an expert grader assessing the relevance of a retrieved document to a user question.

STRICT EVALUATION CRITERIA:
1. Analyze if the provided document contains keywords, facts, or semantic meaning that helps answer the user's question.
2. The document does not need to be a complete answer; if it contains even a small piece of information that contributes toward answering the question, grade it as relevant.
3. If the document is entirely unrelated to the question, grade it as irrelevant.

Respond ONLY with a valid JSON object in this exact format: {{"is_relevant": true}} or {{"is_relevant": false}}

Retrieved Document:
{document}

User Question:
{question}

Response:
"""

QUERY_REWRITE_PROMPT = """
You are an expert AI query optimizer specializing in vector search retrieval.
Your task is to analyze an initial user question that failed to retrieve relevant documents and rewrite it into an optimized version.

CRITICAL GUIDELINES:
1. Strip away all conversational filler, greetings, and meta-commentary (e.g., "Can you tell me...", "I was wondering...", "Please find out...").
2. Focus strictly on the underlying core intent, semantic meaning, and key technical terms.
3. Keep the rewritten query concise, clear, and optimized for a semantic keyword search in a vector store like QDrant.
4. Do NOT answer the question. Only output the rewritten question string.
5. Return ONLY the final optimized query text. Do not include any introduction, quotes, punctuation wrapper, or explanation.

Initial User Question:
{question}

Optimized Vector Search Query:
"""

RESPONSE_GENERATION_PROMPT = """"
You are an expert, fact-based assistant specializing in question-answering tasks.
Your goal is to answer the user's question using strictly the provided context.

CRITICAL GUIDELINES:
1. Rely ONLY on the clear facts directly mentioned in the Context section.
2. Do NOT assume, extrapolate, or bring in outside knowledge not present in the context.
3. Anything not directly mentioned in the context is considered completely untruthful and unsupported.
4. If the context does not contain the answer to the question, respond exactly with:
    "I am sorry, but the provided context does not contain enough information to answer this question."

Context:
{context}

User Question:
{question}

Answer:
"""

HALLUCINATION_CHECK_PROMPT = """
You are an expert auditor assessing whether an AI-generated response is completely grounded in and supported by a set of retrieved documents.

STRICT EVALUATION CRITERIA:
1. Compare the Generated Response against the provided Retrieved Context line by line.
2. The response is considered GROUNDED (not a hallucination) if EVERY single fact, claim, or implication in the response is explicitly stated in the context.
3. The response is considered a HALLUCINATION if it introduces outside knowledge, assumptions, or extrapolations not directly written in the context—even if those facts are accurate in the real world.

Respond ONLY with a valid JSON object in this exact format: {{"is_grounded": true}} or {{"is_grounded": false}}

Do NOT include any introduction, explanation, punctuation, or extra words. Return only the boolean value.

Retrieved Context:
{context}

Generated Response:
{response}

Response:
"""

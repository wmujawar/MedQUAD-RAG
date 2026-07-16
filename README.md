# MedQUAD-RAG

MedQUAD-RAG is a medical-domain Retrieval Augmented Generation (RAG) assistant built around the MedQuAD dataset. The project ingests and chunks healthcare documents, generates embeddings, stores vectors in Qdrant, retrieves relevant context for user queries, and produces grounded responses using configurable LLM providers.

## Project Overview

This repository is structured as a production-ready RAG service with:

1. API layer for chat and query handling.
2. Ingestion pipeline for indexing documents.
3. Pluggable embedding and generation providers.
4. Vector search with Qdrant.
5. Optional observability with Langfuse.
6. Redis-based caching support.
7. Docker Compose setup for local orchestration.

Main application components:

1. src/api: FastAPI routes and versioned endpoints.
2. src/ingestion: Document ingestion and preprocessing.
3. src/embeddings: Embedding providers and cache integration.
4. src/vector_store: Qdrant integration.
5. src/rag: Retrieval and generation pipeline.
6. src/config.py: Centralized settings loaded from environment and secrets.

## Getting Started

Follow these steps in order.

### 1. Start the stack

From the project root, run:

    docker compose up -d

This starts the required local services, including Langfuse dependencies and MinIO.

### 2. Configure environment values

1. Open .env and fill required values.
2. Keep provider/model values aligned with your target setup.
3. If using Qdrant Cloud, set QDRANT_KEY to your cloud API key.
4. If using local Qdrant, QDRANT_KEY can remain empty.

### 3. Langfuse setup

1. Open Langfuse in your browser after services are up.
2. Sign up using the user credentials configured in .env.
3. Create an Organization.
4. Create a Project.
5. Create a new API key.
6. Copy the public key and secret key.
7. Paste the public key into .env for LANGFUSE_PUBLIC_KEY.
8. Create a file named langfuse_secret_key in the secrets directory and paste the secret key into that file.

Generate remaining Langfuse secrets and store them in .env:

1. ENCRYPTION_KEY

	openssl rand -hex 32

2. SALT

	openssl rand -hex 32

3. NEXTAUTH_SECRET

	openssl rand -base64 32

### 4. MinIO setup

1. Open MinIO login page:

	http://localhost:9001/login

2. Login with:
   1. Username from .env value MINIO_ROOT_USER.
   2. Password from .env value MINIO_ROOT_PASSWORD.
3. Create a bucket named `langfuse`.

### 5. Validate configuration

1. Confirm LANGFUSE_PUBLIC_KEY is set in .env.
2. Confirm secrets/langfuse_secret_key exists and contains only the secret key value.
3. Confirm MinIO bucket langfuse exists.
4. Confirm all required containers are healthy.

### 6. Run ingestion and API

After infrastructure is ready:

1. Run document ingestion using scripts in the scripts folder.
2. Start the API service.
3. Send chat/query requests to the API endpoints.

## Notes

1. Never commit real credentials to source control.
2. Keep .env.example as sanitized placeholders only.
3. Store sensitive keys in the secrets directory when supported.

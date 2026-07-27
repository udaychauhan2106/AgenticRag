# Agentic RAG System with LangGraph, Qdrant & Reranking

A stateful Retrieval-Augmented Generation (RAG) application that combines document ingestion, semantic vector search, reranking, conversation-aware query planning, and LLM-based response generation.

The system uses **LangGraph** to orchestrate the query pipeline, **Qdrant** for vector storage and semantic retrieval, **FlashRank** for reranking retrieved documents, **Groq-hosted LLMs** for query planning and response generation, **FastAPI** as the backend API, and **Streamlit** as the user interface.

---

## Overview

Traditional RAG pipelines generally follow a fixed flow:

```text
User Query → Retrieve Documents → Generate Answer
```

This project introduces a stateful orchestration layer using LangGraph:

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Planner Node    │
                         │                     │
                         │ Classify Query and  │
                         │ Rewrite Search Query│
                         └──────────┬──────────┘
                                    │
                                    ▼
                           Conditional Routing
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                Conversational Query     Retrieval Required
                         │                     │
                         │                     ▼
                         │            ┌──────────────────┐
                         │            │ Semantic Search  │
                         │            │     Qdrant       │
                         │            └────────┬─────────┘
                         │                     │
                         │                     ▼
                         │            ┌──────────────────┐
                         │            │    FlashRank     │
                         │            │    Reranking     │
                         │            └────────┬─────────┘
                         │                     │
                         └───────────┬─────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │    Responder Node   │
                          │                     │
                          │ Context + History + │
                          │   Response Generation│
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │     Final Answer    │
                          └─────────────────────┘
```

The planner determines whether a query requires knowledge-base retrieval or can be answered directly using conversation history.

Technical queries are rewritten into retrieval-friendly search queries, embedded, searched against Qdrant, reranked using FlashRank, and passed to the responder for grounded answer generation.

---

## Features

* Document ingestion pipeline
* Recursive directory processing
* Multi-format document parsing
* Configurable text chunking
* Embedding generation
* Vector storage using Qdrant
* Semantic similarity search
* FlashRank-based document reranking
* Conversation-aware query planning
* Retrieval query rewriting
* Conditional routing using LangGraph
* Stateful conversation history using LangGraph checkpointing
* Context-aware response generation
* FastAPI backend
* Streamlit user interface
* Source document display
* Retrieval and pipeline execution status

---

## Tech Stack

| Component           | Technology                                       |
| ------------------- | ------------------------------------------------ |
| Language            | Python                                           |
| Agent Orchestration | LangGraph                                        |
| LLM Framework       | LangChain                                        |
| LLM Provider        | Groq                                             |
| Embeddings          | Google Gemini Embeddings / Sentence Transformers |
| Vector Database     | Qdrant                                           |
| Reranking           | FlashRank                                        |
| Backend API         | FastAPI                                          |
| Frontend            | Streamlit                                        |
| Configuration       | Pydantic Settings                                |
| Package Management  | uv                                               |

---

## Project Structure

```text
RAG-project/
│
├── app/
│   │
│   ├── agents/
│   │   ├── nodes/
│   │   │   ├── planner.py
│   │   │   ├── retriever.py
│   │   │   └── responder.py
│   │   │
│   │   ├── graph.py
│   │   └── state.py
│   │
│   ├── ingestion/
│   │   ├── processor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── parsers/
│   │
│   ├── config/
│   │
│   └── main.py
│
├── ui/
│   └── app.py
│
├── DATA/
│
├── processed_data/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# System Architecture

The application consists of two independent pipelines:

1. Document ingestion pipeline
2. Query-time RAG pipeline

---

## 1. Document Ingestion Pipeline

The ingestion pipeline processes documents and stores their vector representations in Qdrant.

```text
Documents
    │
    ▼
File Discovery
    │
    ▼
Document Parser
    │
    ▼
Extracted Text
    │
    ▼
Text Chunking
    │
    ▼
Embedding Generation
    │
    ▼
Vectors + Metadata
    │
    ▼
Qdrant Vector Database
```

### Processing Steps

#### File Discovery

The ingestion pipeline recursively scans the configured data directory for supported documents.

#### Document Parsing

Files are processed using format-specific parsers to extract textual content and metadata.

#### Text Chunking

Extracted documents are divided into smaller chunks suitable for semantic retrieval and LLM context construction.

#### Embedding Generation

Each chunk is converted into a dense vector representation using the configured embedding model.

#### Vector Storage

The embeddings are stored in Qdrant together with metadata such as source information and chunk content.

---

## 2. Query-Time RAG Pipeline

User queries are processed through a LangGraph state machine.

```text
START
  │
  ▼
Planner
  │
  ▼
Conditional Router
  │
  ├──────── Conversational Query ─────────────┐
  │                                           │
  ▼                                           │
Retriever                                     │
  │                                           │
  ▼                                           │
Query Embedding                               │
  │                                           │
  ▼                                           │
Qdrant Vector Search                          │
  │                                           │
  ▼                                           │
FlashRank Reranking                           │
  │                                           │
  ▼                                           │
Retrieved Context                             │
  │                                           │
  └───────────────────┬───────────────────────┘
                      │
                      ▼
                   Responder
                      │
                      ▼
                 Final Answer
                      │
                      ▼
                     END
```

---

## LangGraph State

Application state is shared between nodes during graph execution.

The state contains information such as:

```text
messages

current_query

documents

plan

status

final_answer
```

Each LangGraph node receives the current state and returns partial state updates.

LangGraph merges those updates before executing the next node.

Conversation messages use an additive reducer so new messages are appended to the existing conversation history.

---

## Planner Node

The planner is the first node executed for every user request.

Responsibilities:

* Analyze the current user query
* Inspect conversation history
* Determine whether retrieval is necessary
* Rewrite technical questions into retrieval-friendly search queries
* Route conversational queries directly to response generation

Example:

```text
User Query

"Can you explain how checkpointing works in LangGraph?"
```

Planner output:

```text
LangGraph checkpointing and conversation persistence
```

The rewritten query is then passed to the retrieval pipeline.

For conversational messages such as:

```text
Hello!
```

the planner marks the query as conversational and skips document retrieval.

---

## Retrieval Pipeline

The retrieval node performs a two-stage retrieval process.

### Stage 1: Semantic Vector Search

The rewritten query is converted into an embedding.

```text
Search Query
      │
      ▼
Embedding Model
      │
      ▼
Query Vector
      │
      ▼
Qdrant
      │
      ▼
Top-K Candidate Documents
```

Qdrant retrieves semantically similar chunks from the vector database.

### Stage 2: Document Reranking

The retrieved candidates are passed to FlashRank.

```text
Query
  +
Candidate Documents
        │
        ▼
    FlashRank
        │
        ▼
Relevance Scores
        │
        ▼
Top Ranked Documents
```

Reranking improves the ordering of retrieved documents before they are passed to the language model.

---

## Responder Node

The responder generates the final answer.

For retrieval-based queries, the prompt contains:

```text
Retrieved Context

+

Conversation History

+

User Question
```

For conversational queries, the responder uses conversation history without performing vector retrieval.

The generated assistant response is appended to the LangGraph message history and stored in the final graph state.

---

## Conversation Memory

The application uses LangGraph checkpointing with a thread identifier.

```text
thread_id
    │
    ▼
LangGraph Checkpointer
    │
    ▼
Conversation State
```

Requests using the same `thread_id` can continue the same conversation.

Requests using different thread identifiers maintain independent conversation state.

The current implementation uses an in-memory checkpointer, so conversation state is lost when the application process restarts.

---

# API

The backend is implemented using FastAPI.

## Query Endpoint

```text
POST /query
```

### Example Request

```json
{
    "question": "Explain how transformers use self-attention.",
    "thread_id": "example-session-123"
}
```

### Example Response

```json
{
    "answer": "Self-attention allows each token in a sequence to...",
    "thought_process": [
        "Query classified",
        "Retrieval required",
        "Context Retrieved",
        "Answer Generated"
    ],
    "status": "Complete",
    "sources": []
}
```

---

## Graph Visualization Endpoint

```text
GET /graph
```

Returns a Mermaid-generated visualization of the compiled LangGraph workflow.

---



# Example Query Flow

Consider the following user query:

```text
"How does self-attention work in transformers?"
```

### Step 1: FastAPI

The API receives:

```json
{
    "question": "How does self-attention work in transformers?",
    "thread_id": "session-123"
}
```

An initial LangGraph state is created.

### Step 2: Planner

The planner analyzes the conversation and generates a retrieval query:

```text
transformer self-attention mechanism
```

### Step 3: Routing

The conditional router detects that retrieval is required and sends execution to the retriever node.

### Step 4: Query Embedding

The retrieval query is converted into an embedding vector.

### Step 5: Vector Search

Qdrant retrieves the most semantically similar document chunks.

### Step 6: Reranking

FlashRank reranks the candidate documents according to their relevance to the query.

### Step 7: Response Generation

The responder receives:

```text
Top Ranked Documents

+

Conversation History

+

Original User Question
```

The LLM generates the final response.

### Step 8: State Update

The assistant response is appended to the conversation history.

### Step 9: API Response

FastAPI returns the generated answer, pipeline execution information, and retrieved sources to the client.

---

# Current Limitations

The project is currently a learning and experimentation project rather than a production-ready RAG service.

Known limitations include:

* Conversation checkpoints are stored in process memory and are lost after application restarts.
* The system does not currently perform retrieval relevance thresholding.
* Retrieved answers do not include verified inline citations.
* Long conversation histories are not automatically summarized or trimmed.
* Automated RAG evaluation and retrieval benchmarks are not yet implemented.
* Deployment configuration and CI/CD pipelines are not yet included.

---

# Future Improvements

Planned improvements include:

* Structured planner outputs using Pydantic models
* Persistent LangGraph checkpoint storage
* Retrieval relevance filtering
* Hybrid dense and sparse retrieval
* Source citations in generated answers
* Conversation history summarization
* Streaming responses
* Retrieval and generation evaluation pipelines
* LangSmith tracing and observability
* Automated testing
* Docker-based deployment
* CI/CD pipeline
* Health and readiness endpoints

---

# Learning Objectives

This project was built to gain practical experience with:

* Designing end-to-end RAG applications
* Building document ingestion pipelines
* Working with vector databases
* Understanding semantic retrieval
* Implementing reranking pipelines
* Designing stateful AI workflows
* Using LangGraph for orchestration
* Building APIs around LLM applications
* Maintaining conversation state
* Connecting AI backends with user interfaces



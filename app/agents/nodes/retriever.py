import logfire
import time
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents_with_indices

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        timings: dict[str, float] = {}
        retrieval_started = time.perf_counter()
        raw_results = search_enterprise_knowledge(query, limit=15, timings=timings)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        rerank_started = time.perf_counter()
        with logfire.span("Semantic Reranking"):
            reranked = rerank_documents_with_indices(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
        reranking_ms = (time.perf_counter() - rerank_started) * 1000
            
        selected_results = [raw_results[index] for index, _ in reranked]
        formatted_docs = [f"CONTENT: {doc['content']}" for doc in selected_results]
    
    return {
        "documents": formatted_docs,
        "retrieved_chunk_ids": [doc["id"] for doc in selected_results],
        "retrieved_document_ids": [doc["source"] for doc in selected_results],
        "embedding_ms": timings.get("embedding_ms", 0),
        "vector_search_ms": timings.get("vector_search_ms", 0),
        "retrieval_ms": retrieval_ms,
        "reranking_ms": reranking_ms,
        "retrieved_candidates": len(raw_results),
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
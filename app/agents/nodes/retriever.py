import logging

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

logger = logging.getLogger(__name__)


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    logger.info("Searching Qdrant for: %s", query)
    raw_results = search_enterprise_knowledge(query, limit=15)
    logger.info("Retrieved %d candidates from vector database", len(raw_results))

    doc_contents = [doc["content"] for doc in raw_results]
    reranked_contents = rerank_documents(query, doc_contents, top_n=5)
    logger.info("Reranking complete; selected %d chunks", len(reranked_contents))

    formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
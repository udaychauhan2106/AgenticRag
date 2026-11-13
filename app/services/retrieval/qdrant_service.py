import logfire
import time
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embeddings import embed_query


client = QdrantClient(
    url=settings.QUADRANT_URL,
    api_key=settings.QUADRANT_API_KEY
)

def search_enterprise_knowledge(
    query: str, limit: int = 8, timings: dict[str, float] | None = None
):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        started = time.perf_counter()
        query_vector = embed_query(query)
        if timings is not None:
            timings["embedding_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        response = client.query_points(
            collection_name=settings.QUADRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True 
        )
        if timings is not None:
            timings["vector_search_ms"] = (time.perf_counter() - started) * 1000

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "id": str(res.id),
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": res.score
            })
        
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise
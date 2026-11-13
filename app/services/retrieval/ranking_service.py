import time
import logging

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logger.info("Initializing FlashRank model locally")
        try:
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            logger.exception("Could not load FlashRank from its configured cache; using the default cache")
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.

    """
    if not documents:
        return []

    start_time = time.time()
    logger.info("Reranking %d documents with FlashRank", len(documents))

    try:
        ranker = _get_ranker()
        
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logger.info("FlashRank reranking finished in %.2fs; top score: %s", duration, top_score)
        
        return reranked_docs

    except Exception:
        logger.exception("FlashRank reranking failed; returning original retrieval order")
        return documents[:top_n]
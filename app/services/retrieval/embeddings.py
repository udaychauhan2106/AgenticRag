import time
import random
import logfire
 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app import settings
 
BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768
 
_active_model = None
_model_type: str | None = None
 
 
class QuotaExhausted(Exception):
    """Raised when Gemini reports a daily/hard quota exhaustion (not a transient rate limit).
    Callers should stop using Gemini for the remainder of the run rather than retrying."""
    pass
 
 
def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim)")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None
 
 
def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim)")
    return SentenceTransformer("all-mpnet-base-v2")
 
 
def _init():
    global _active_model, _model_type
    if _active_model is not None:
        return
    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"
 
 
def reset():
    """Force re-probe on next call. Useful for tests or after switching API keys."""
    global _active_model, _model_type
    _active_model = None
    _model_type = None
 
 
def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Triggers _init() if needed."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def get_embedding_model_name(initialize: bool = True) -> str:
    """Return the model currently selected for embeddings."""
    if initialize:
        _init()
    if _model_type is None:
        return "not_initialized"
    return (
        "models/gemini-embedding-2-preview"
        if _model_type == "gemini"
        else "sentence-transformers/all-mpnet-base-v2"
    )
 
 
def _classify_error(err: str) -> str:
    """Classify a Gemini error string into: daily_quota | rate_limit | other."""
    err = err.lower()
    if "resource_exhausted" in err or "quota" in err:
        if "day" in err or "daily" in err:
            return "daily_quota"
        return "rate_limit"
    if "429" in err:
        return "rate_limit"
    return "other"
 
 
def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                kind = _classify_error(str(e))
 
                if kind == "daily_quota":
                    logfire.error(f"Gemini daily quota exhausted: {e}")
                    raise QuotaExhausted(str(e)) from e
 
                if kind == "rate_limit" and attempt < 3:
                    wait = min(2 ** attempt + random.uniform(0, 1), 30)
                    logfire.warning(
                        f"Gemini rate limit hit - retrying in {wait:.1f}s "
                        f"(attempt {attempt + 1}/4)"
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()
 
 
def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query], show_progress_bar=False)[0].tolist()
 
 
def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
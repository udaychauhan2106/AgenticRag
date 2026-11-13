import logfire
import os
import time
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

from fastapi import FastAPI, HTTPException, Response
from app.agents.graph import rag_agent
from app.guardrails.rails import initialize_rails, guard
from app.config import settings
from app.metrics import (
    estimate_query_cost_usd,
    lexical_token_count,
    record_query_run,
    runtime_metrics_summary,
)
from app.services.retrieval.embeddings import get_embedding_model_name


from pydantic import BaseModel
from typing import Optional

app=FastAPI(title="Enterprise Agentic RAG API")

@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str]="default_user"


@app.get("/")
def home():
    return {"message":"Enterprise LangGrpah RAG API is live"}


@app.get("/metrics")
def get_metrics():
    return runtime_metrics_summary()


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes=rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes,media_type="image/png")
    
    except Exception as e:
        return {"error" : f"Could not generate graph image: {e}"}
    
@app.post("/query")
def query(request:QueryRequest):
    """
    Executes the Langgrpah RAG flow with memory
    using a POST request.
    """
    q=request.q
    thread_id=request.thread_id

    initial_state={
        "messages":[{"role":"user" , "content":q}],
        "model_usage": [],
        "current_query":q,
        "documents":[],
        "retrieved_chunk_ids": [],
        "retrieved_document_ids": [],
        "embedding_ms": 0.0,
        "vector_search_ms": 0.0,
        "retrieval_ms": 0.0,
        "reranking_ms": 0.0,
        "planner_ms": 0.0,
        "generation_ms": 0.0,
        "retrieved_candidates": 0,
        "plan":["Start"],
        "status":"Initializing Graph..."
    }

    config = {"configurable": {"thread_id": thread_id}}
    request_started = time.perf_counter()
    metrics: dict = {
        "guardrails_ms": 0.0,
        "planner_ms": 0.0,
        "embedding_ms": 0.0,
        "vector_search_ms": 0.0,
        "retrieval_ms": 0.0,
        "reranking_ms": 0.0,
        "generation_ms": 0.0,
        "model_usage": [],
        "embedding_tokens_estimate": 0,
        "retrieved_chunks": 0,
        "retrieved_candidates": 0,
    }
    succeeded = False
    timed_out = False
    failure_stage: str | None = "guardrails"
    try:
        stage_started = time.perf_counter()
        rail_fired, rail_response, guard_usage = guard(q)
        metrics["model_usage"].extend(guard_usage)
        metrics["guardrails_ms"] = (time.perf_counter() - stage_started) * 1000
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            succeeded = True
            failure_stage = None
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        failure_stage = "agent_graph"
        final_output = rag_agent.invoke(initial_state, config=config)
        for field in (
            "planner_ms",
            "embedding_ms",
            "vector_search_ms",
            "retrieval_ms",
            "reranking_ms",
            "generation_ms",
        ):
            metrics[field] = float(final_output.get(field, 0) or 0)
        metrics["model_usage"] = final_output.get("model_usage", [])
        metrics["model_usage"] = guard_usage + metrics["model_usage"]
        metrics["embedding_tokens_estimate"] = (
            lexical_token_count(final_output.get("current_query", ""))
            if final_output.get("retrieval_ms")
            else 0
        )
        metrics["retrieved_chunks"] = len(final_output.get("retrieved_chunk_ids", []))
        metrics["retrieved_candidates"] = int(final_output.get("retrieved_candidates", 0) or 0)
        succeeded = True
        failure_stage = None
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        timed_out = isinstance(e, TimeoutError) or "timeout" in str(e).lower()
        logfire.error(f"❌ Backend Execution Failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="The AI backend could not process the request. Check backend logs for details.",
        ) from e
    finally:
        model_usage = metrics["model_usage"]
        try:
            record_query_run(
                end_to_end_ms=(time.perf_counter() - request_started) * 1000,
                guardrails_ms=metrics["guardrails_ms"],
                planner_ms=metrics["planner_ms"],
                embedding_ms=metrics["embedding_ms"],
                vector_search_ms=metrics["vector_search_ms"],
                retrieval_ms=metrics["retrieval_ms"],
                reranking_ms=metrics["reranking_ms"],
                generation_ms=metrics["generation_ms"],
                input_tokens=sum(int(item.get("input_tokens", 0)) for item in model_usage),
                output_tokens=sum(int(item.get("output_tokens", 0)) for item in model_usage),
                estimated_cost_usd=estimate_query_cost_usd(
                    model_usage,
                    get_embedding_model_name(initialize=False),
                    metrics["embedding_tokens_estimate"],
                ),
                success=succeeded,
                timed_out=timed_out,
                failure_stage=failure_stage,
                retrieved_chunks=metrics["retrieved_chunks"],
                retrieved_candidates=metrics["retrieved_candidates"],
                model_usage=model_usage,
                embedding_tokens_estimate=metrics["embedding_tokens_estimate"],
                embedding_model=get_embedding_model_name(initialize=False),
                collection_name=settings.QUADRANT_COLLECTION,
            )
        except Exception:
            logfire.exception("Could not persist request metrics")

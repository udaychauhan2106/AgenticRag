import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from app.agents.graph import rag_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Agentic RAG API")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"

@app.get("/")
def home():
    return {"message":"Enterprise LangGrpah RAG API is live"}


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
def query(request: QueryRequest):
    """
    Executes the Langgrpah RAG flow with memory
    using a POST request.
    """
    q=request.q
    thread_id=request.thread_id

    initial_state={
        "messages":[{"role":"user" , "content":q}],
        "current_query":q,
        "documents":[],
        "plan":["Start"],
        "status":"Initializing Graph..."
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as exc:
        logger.exception("Backend execution failed")
        raise HTTPException(
            status_code=502,
            detail="The AI backend could not process the request. Check backend logs for details.",
        ) from exc

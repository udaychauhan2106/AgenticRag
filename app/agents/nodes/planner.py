from app.agents.state import AgentState
from app.config import settings
from langchain_groq import ChatGroq
import logfire
import time

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_PLANNER_MODEL,
    temperature=0,
)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on ENTIRE conversation.
    """

    history=""
    for msg in state["messages"][:-1]:
        role="User" if msg["role"]=="user" else "Assistant"
        history+=f"{role}:{msg['content']}\n"
    
    user_message=state["messages"][-1]["content"] if state["messages"] else ""

    prompt=f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    with logfire.span("Planner Decision"):
        started = time.perf_counter()
        response = llm.invoke(prompt)
        planner_ms = (time.perf_counter() - started) * 1000
        decision = response.content.strip()
        logfire.info(f"Intent identified :{decision}")

    usage = response.usage_metadata or {}
    metrics = {
        "planner_ms": planner_ms,
        "model_usage": [{
            "model": settings.GROQ_PLANNER_MODEL,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }],
    }

    if decision =="CONVERSATIONAL":
        return {
            **metrics,
            "current_query" : "CONVERSATIONAL",
            "status":"Handling conversationally (using memory)...",
            "plan":["Intent:Conversational/Memory","Retrieval: Skipped"]
        }
    return {
        **metrics,
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
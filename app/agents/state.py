from typing import TypedDict ,List ,Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    model_usage: Annotated[List[dict], operator.add]
    current_query :str
    documents:List[str]
    retrieved_chunk_ids: List[str]
    retrieved_document_ids: List[str]
    embedding_ms: float
    vector_search_ms: float
    retrieval_ms: float
    reranking_ms: float
    planner_ms: float
    generation_ms: float
    retrieved_candidates: int
    plan :List[str]
    status : str
    final_answer: str

    
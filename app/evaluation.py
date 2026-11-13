import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient

from app.config import settings
from app.metrics import runtime_metrics_summary
from app.services.retrieval.embeddings import embed_query, get_embedding_model_name
from app.services.retrieval.ranking_service import (
    RERANKER_MODEL,
    rerank_documents_with_indices,
)


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "eval" / "golden.json"
DEFAULT_PROCESSED_DATA = Path(__file__).resolve().parents[1] / "processed_data" / "true"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "evaluation.json"
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "what", "when", "which", "with",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def retrieval_scores(
    ranked_ids: list[str], relevant_ids: set[str], k: int = 5
) -> dict[str, float | None]:
    """Compute binary-relevance Recall@K, Precision@K, MRR@K, and nDCG@K."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return {
            f"recall@{k}": None,
            f"precision@{k}": None,
            f"mrr@{k}": None,
            f"ndcg@{k}": None,
        }

    top_k = list(dict.fromkeys(ranked_ids))[:k]
    hits = [1 if item in relevant_ids else 0 for item in top_k]
    relevant_found = sum(hits)
    first_rank = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        f"recall@{k}": relevant_found / len(relevant_ids),
        f"precision@{k}": relevant_found / k,
        f"mrr@{k}": 1 / first_rank if first_rank else 0.0,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _aggregate(scores: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = sorted({key for score in scores for key in score})
    return {
        key: (
            statistics.fmean(
                float(score[key]) for score in scores if score.get(key) is not None
            )
            if any(score.get(key) is not None for score in scores)
            else None
        )
        for key in keys
    }


def _document_id(point: Any) -> str:
    return str((point.payload or {}).get("source", "Unknown"))


def _chunk_id(point: Any) -> str:
    payload = point.payload or {}
    return f"{payload.get('source', 'Unknown')}#{payload.get('chunk_index', '')}"


def _unique_documents(points: list[Any], k: int) -> list[Any]:
    unique = []
    seen: set[str] = set()
    for point in points:
        document_id = _document_id(point)
        if document_id not in seen:
            unique.append(point)
            seen.add(document_id)
        if len(unique) == k:
            break
    return unique


def _corpus_fingerprint(processed_data: Path) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    document_count = 0
    chunk_count = 0
    token_count = 0
    file_types: dict[str, int] = {}

    for path in sorted(processed_data.glob("*.json")):
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(raw)
        record = json.loads(raw)
        document_count += 1
        extension = Path(record.get("filename", path.stem)).suffix.lower() or "(none)"
        file_types[extension] = file_types.get(extension, 0) + 1
        chunks = record.get("chunks", [])
        chunk_count += len(chunks)
        token_count += sum(len(_tokens(chunk)) for chunk in chunks)

    return digest.hexdigest(), {
        "processed_documents": document_count,
        "processed_chunks": chunk_count,
        "lexical_token_count": token_count,
        "token_count_method": "Unicode word-token estimate; not provider tokenizer tokens",
        "file_types": dict(sorted(file_types.items())),
    }


def _evaluation_code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    source_files = (
        Path(__file__),
        root / "app" / "config.py",
        root / "app" / "agents" / "nodes" / "planner.py",
        root / "app" / "agents" / "nodes" / "responder.py",
        root / "app" / "services" / "retrieval" / "embeddings.py",
        root / "app" / "services" / "retrieval" / "ranking_service.py",
    )
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_relevance_labels(
    examples: list[dict[str, Any]], processed_data: Path
) -> None:
    documents: set[str] = set()
    chunks: set[str] = set()
    for path in sorted(processed_data.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        filename = record.get("filename")
        if not filename:
            raise ValueError(f"{path} does not declare a filename")
        documents.add(filename)
        chunks.update(
            f"{filename}#{index}"
            for index, _ in enumerate(record.get("chunks", []))
        )

    for example in examples:
        unknown_docs = set(example["relevant_document_ids"]) - documents
        unknown_chunks = set(example["relevant_chunk_ids"]) - chunks
        if unknown_docs or unknown_chunks:
            raise ValueError(
                f"{example['id']} labels are not present in the selected trusted corpus: "
                f"documents={sorted(unknown_docs)}, chunks={sorted(unknown_chunks)}"
            )


def _collection_fingerprint(qdrant: QdrantClient) -> tuple[str, int, set[str]]:
    digest = hashlib.sha256()
    point_count = 0
    source_types: set[str] = set()
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=settings.QUADRANT_COLLECTION,
            offset=offset,
            limit=1000,
            with_payload=["source", "source_type", "chunk_index", "text"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source_types.add(str(payload.get("source_type", "")))
            digest.update(str(point.id).encode("utf-8"))
            digest.update(str(payload.get("source", "")).encode("utf-8"))
            digest.update(str(payload.get("chunk_index", "")).encode("utf-8"))
            digest.update(str(payload.get("text", "")).encode("utf-8"))
            point_count += 1
        if offset is None:
            break
    return digest.hexdigest(), point_count, source_types


def _load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("examples"), list):
        raise ValueError("Evaluation dataset must be an object with an examples array")
    if not data.get("version"):
        raise ValueError("Evaluation dataset must declare a version")

    required = {
        "id", "question", "expected_answer", "answerable",
        "relevant_document_ids", "relevant_chunk_ids",
    }
    ids: set[str] = set()
    for example in data["examples"]:
        missing = required - example.keys()
        if missing:
            raise ValueError(
                f"Evaluation example is missing required fields: {sorted(missing)}"
            )
        if example["id"] in ids:
            raise ValueError(f"Duplicate evaluation example id: {example['id']}")
        ids.add(example["id"])
        if not isinstance(example["answerable"], bool):
            raise ValueError(f"{example['id']}: answerable must be a boolean")
        if not isinstance(example["relevant_document_ids"], list) or not isinstance(
            example["relevant_chunk_ids"], list
        ):
            raise ValueError(f"{example['id']}: relevance labels must be arrays")
    return data, data["examples"], hashlib.sha256(raw).hexdigest()


def _quality_proxies(
    answer: str,
    reference: str,
    context: str,
    answerable: bool,
) -> dict[str, float | None]:
    answer_tokens = [token for token in _tokens(answer) if token not in _STOPWORDS]
    reference_tokens = [token for token in _tokens(reference) if token not in _STOPWORDS]
    context_tokens = set(_tokens(context))

    if not answerable:
        abstention_terms = (
            "not enough information", "cannot determine", "can't determine",
            "not provided", "does not provide", "do not have enough",
            "don't have enough", "cannot answer",
        )
        lower_answer = answer.lower()
        abstention = float(any(term in lower_answer for term in abstention_terms))
        return {
            "reference_token_f1": None,
            "context_lexical_support_proxy": None,
            "unanswerable_abstention": abstention,
        }

    answer_counts = {token: answer_tokens.count(token) for token in set(answer_tokens)}
    reference_counts = {
        token: reference_tokens.count(token) for token in set(reference_tokens)
    }
    overlap = sum(min(count, reference_counts.get(token, 0)) for token, count in answer_counts.items())
    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = (
        sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    return {
        "reference_token_f1": f1,
        "context_lexical_support_proxy": support,
        "unanswerable_abstention": None,
    }


def _judge_answer(
    question: str, expected: str, answer: str, context: str
) -> dict[str, Any]:
    judge = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    prompt = f"""Evaluate this RAG answer against the question, reference answer, and retrieved context.
Return only JSON with numeric scores from 1 to 5 and a short explanation for each:
{{"faithfulness":{{"score":1,"reason":"..."}},
 "correctness":{{"score":1,"reason":"..."}},
 "context_relevance":{{"score":1,"reason":"..."}}}}

Question: {question}
Reference answer: {expected}
Retrieved context: {context}
Generated answer: {answer}
"""
    content = judge.invoke(prompt).content
    if not isinstance(content, str):
        raise ValueError("LLM judge returned non-text content")
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM judge did not return a JSON object")
    result = json.loads(match.group())
    for metric in ("faithfulness", "correctness", "context_relevance"):
        score = result[metric]["score"]
        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(f"LLM judge returned invalid {metric} score")
    return result


def _load_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    reviews: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        review = json.loads(line)
        example_id = review.get("id")
        if not example_id or example_id in reviews:
            raise ValueError(f"{path}:{line_number}: missing or duplicate review id")
        for key in ("faithfulness", "correctness", "context_relevance"):
            score = review.get(key)
            if score is not None and (
                not isinstance(score, (int, float)) or not 1 <= score <= 5
            ):
                raise ValueError(f"{path}:{line_number}: {key} must be 1..5 or null")
        reviews[example_id] = review
    return reviews


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Path = DEFAULT_REPORT,
    *,
    k: int = 5,
    candidate_k: int = 15,
    processed_data: Path = DEFAULT_PROCESSED_DATA,
    review_path: Path | None = None,
    use_judge: bool = False,
) -> dict[str, Any]:
    if k <= 0 or candidate_k < k:
        raise ValueError("Require candidate_k >= k > 0")

    dataset, examples, dataset_hash = _load_dataset(dataset_path)
    reviews = _load_reviews(review_path)
    unknown_reviews = set(reviews) - {example["id"] for example in examples}
    if unknown_reviews:
        raise ValueError(f"Reviews reference unknown examples: {sorted(unknown_reviews)}")
    corpus_hash, corpus_summary = _corpus_fingerprint(processed_data)
    _validate_relevance_labels(examples, processed_data)
    qdrant = QdrantClient(url=settings.QUADRANT_URL, api_key=settings.QUADRANT_API_KEY)
    collection = qdrant.get_collection(settings.QUADRANT_COLLECTION)
    collection_hash, collection_count, source_types = _collection_fingerprint(qdrant)
    if collection_count != corpus_summary["processed_chunks"]:
        raise ValueError(
            "The selected processed-data corpus does not match the Qdrant collection: "
            f"{corpus_summary['processed_chunks']} processed chunks vs "
            f"{collection_count} indexed points."
        )
    if not source_types:
        raise ValueError("The baseline Qdrant collection is empty.")
    if source_types - {"true"}:
        raise ValueError(
            "The baseline collection must contain trusted source_type='true' points only; "
            f"found {sorted(source_types)}. Keep noisy data in a separate experiment collection."
        )

    example_results = []
    semantic_scores = []
    reranked_scores = []
    answer_scores = []
    judge_scores: list[dict[str, float | None]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    for example in examples:
        vector = embed_query(example["question"])
        retrieval = qdrant.query_points(
            collection_name=settings.QUADRANT_COLLECTION,
            query=vector,
            limit=candidate_k,
            with_payload=True,
        )
        candidates = list(retrieval.points)
        semantic_ranked = candidates[:k]
        contents = [str((point.payload or {}).get("text", "")) for point in candidates]
        reranked_indices = rerank_documents_with_indices(
            example["question"], contents, top_n=candidate_k
        )
        reranked_all = [candidates[index] for index, _ in reranked_indices]
        reranked = reranked_all[:k]
        semantic_document_ranked = _unique_documents(candidates, k)
        reranked_document_ranked = _unique_documents(reranked_all, k)

        relevant_docs = set(example["relevant_document_ids"])
        relevant_chunks = set(example["relevant_chunk_ids"])
        semantic_doc_scores = retrieval_scores(
            [_document_id(point) for point in semantic_document_ranked], relevant_docs, k
        )
        reranked_doc_scores = retrieval_scores(
            [_document_id(point) for point in reranked_document_ranked], relevant_docs, k
        )
        semantic_chunk_scores = retrieval_scores(
            [_chunk_id(point) for point in semantic_ranked], relevant_chunks, k
        )
        reranked_chunk_scores = retrieval_scores(
            [_chunk_id(point) for point in reranked], relevant_chunks, k
        )
        semantic_scores.append(semantic_doc_scores)
        reranked_scores.append(reranked_doc_scores)

        context = "\n\n".join(
            str((point.payload or {}).get("text", "")) for point in reranked
        )
        answer = ""
        from app.agents.nodes.responder import generate_node

        generated = generate_node(
            {
                "messages": [{"role": "user", "content": example["question"]}],
                "current_query": example["question"],
                "documents": [
                    f"CONTENT: {(point.payload or {}).get('text', '')}"
                    for point in reranked
                ],
                "plan": ["Evaluation"],
                "status": "Evaluation",
            }
        )
        answer = str(generated["final_answer"])

        quality = _quality_proxies(
            answer,
            example["expected_answer"],
            context,
            example["answerable"],
        )
        retrieved_doc_ids = [_document_id(point) for point in reranked_document_ranked]
        retrieved_chunk_ids = [_chunk_id(point) for point in reranked]
        context_document_ids = list(dict.fromkeys(_document_id(point) for point in reranked))
        context_relevance = (
            sum(doc_id in relevant_docs for doc_id in context_document_ids)
            / len(context_document_ids)
            if relevant_docs and context_document_ids
            else None
        )
        quality["context_relevance_precision"] = context_relevance
        answer_scores.append(quality)

        result = {
            "id": example["id"],
            "question": example["question"],
            "answerable": example["answerable"],
            "expected_answer": example["expected_answer"],
            "generated_answer": answer,
            "retrieval": {
                "semantic_only": {
                    "document_metrics": semantic_doc_scores,
                    "chunk_metrics": semantic_chunk_scores,
                    "ranked_document_ids": [
                        _document_id(point) for point in semantic_document_ranked
                    ],
                    "ranked_chunk_ids": [_chunk_id(point) for point in semantic_ranked],
                },
                "semantic_plus_flashrank": {
                    "document_metrics": reranked_doc_scores,
                    "chunk_metrics": reranked_chunk_scores,
                    "ranked_document_ids": retrieved_doc_ids,
                    "ranked_chunk_ids": retrieved_chunk_ids,
                },
            },
            "answer_quality_proxies": {
                **quality,
                "context_relevance_precision": context_relevance,
                "scoring_note": (
                    "Reference token F1 and lexical context support are transparent "
                    "lexical proxies, not semantic proof of correctness or faithfulness."
                ),
            },
            "human_review": reviews.get(
                example["id"],
                {
                    "faithfulness": None,
                    "correctness": None,
                    "context_relevance": None,
                    "reviewer": None,
                },
            ),
        }
        if use_judge:
            judge_result = _judge_answer(
                example["question"], example["expected_answer"], answer, context
            )
            result["llm_judge"] = judge_result
            judge_scores.append({
                metric: float(judge_result[metric]["score"])
                for metric in ("faithfulness", "correctness", "context_relevance")
            })
        example_results.append(result)

    source_digest_files = (
        [path.name for path in sorted(processed_data.glob("*.json"))]
        if processed_data.exists()
        else []
    )
    repository_root = Path(__file__).resolve().parents[1]
    try:
        processed_data_display_path = processed_data.resolve().relative_to(
            repository_root
        ).as_posix()
    except ValueError:
        processed_data_display_path = processed_data.name

    report = {
        "report_version": "1",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset.get("name", dataset_path.stem),
            "version": dataset["version"],
            "sha256": dataset_hash,
            "examples": len(examples),
        },
        "reproducibility": {
            "evaluation_code_sha256": _evaluation_code_fingerprint(),
            "models": {
                "guardrails": settings.GROQ_PLANNER_MODEL,
                "planner": settings.GROQ_PLANNER_MODEL,
                "answer_generator": settings.GROQ_MODEL,
                "embedding": get_embedding_model_name(),
                "optional_judge": settings.GROQ_MODEL if use_judge else None,
            },
            "retrieval_configuration": {
                "collection": settings.QUADRANT_COLLECTION,
                "collection_status": str(collection.status),
                "vector_dimension": collection.config.params.vectors.size,
                "vector_distance": str(collection.config.params.vectors.distance),
                "candidate_k": candidate_k,
                "evaluation_k": k,
                "reranker": RERANKER_MODEL,
                "reranker_returned_candidates": candidate_k,
                "answer_context_chunks": k,
                "planner_temperature": 0,
                "answer_temperature": 0,
            },
            "corpus": {
                "processed_data_path": processed_data_display_path,
                "sha256": corpus_hash,
                "collection_point_count": collection_count,
                "collection_sha256": collection_hash,
                "collection_source_types": sorted(source_types),
                "processed_files": source_digest_files,
                **corpus_summary,
            },
        },
        "retrieval": {
            "semantic_only_document_metrics": _aggregate(semantic_scores),
            "semantic_plus_flashrank_document_metrics": _aggregate(reranked_scores),
            "semantic_only_chunk_metrics": _aggregate([
                result["retrieval"]["semantic_only"]["chunk_metrics"]
                for result in example_results
            ]),
            "semantic_plus_flashrank_chunk_metrics": _aggregate([
                result["retrieval"]["semantic_plus_flashrank"]["chunk_metrics"]
                for result in example_results
            ]),
            "comparison": "Both modes use the same semantic candidate set; FlashRank reorders its top candidates.",
        },
        "answer_quality": {
            "deterministic_proxy_averages": _aggregate(answer_scores),
            "human_review_scores_are_separate": True,
            "llm_judge_enabled": use_judge,
            "llm_judge_is_not_ground_truth": True,
            "llm_judge_average_scores_1_to_5": _aggregate(judge_scores),
            "human_review_scores_1_to_5": _aggregate([
                {
                    key: (
                        float(result["human_review"][key])
                        if result["human_review"].get(key) is not None
                        else None
                    )
                    for key in ("faithfulness", "correctness", "context_relevance")
                }
                for result in example_results
            ]),
            "human_reviewed_examples": sum(
                any(
                    result["human_review"].get(key) is not None
                    for key in ("faithfulness", "correctness", "context_relevance")
                )
                for result in example_results
            ),
        },
        "runtime_metrics": runtime_metrics_summary(),
        "examples": example_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED_DATA)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--judge", action="store_true", help="Add optional Groq LLM-judge scores")
    args = parser.parse_args(argv)
    report = run_evaluation(
        args.dataset,
        args.output,
        k=args.k,
        candidate_k=args.candidate_k,
        processed_data=args.processed_data,
        review_path=args.review_file,
        use_judge=args.judge,
    )

    print(f"Evaluation report: {args.output}")
    print(
        f"Dataset: {report['dataset']['name']} v{report['dataset']['version']} "
        f"({report['dataset']['examples']} examples)"
    )
    print(f"Corpus: {report['reproducibility']['corpus']['collection_point_count']} Qdrant chunks")
    print("Retrieval metrics (document-level; mean across answerable examples):")
    for name in (
        "semantic_only_document_metrics",
        "semantic_plus_flashrank_document_metrics",
    ):
        scores = report["retrieval"][name]
        print(f"  {name}: " + ", ".join(
            f"{metric}={value:.3f}" if value is not None else f"{metric}=n/a"
            for metric, value in scores.items()
        ))
    print("Answer-quality values are labeled lexical proxies; human reviews are reported separately.")
    if args.judge:
        print("LLM-judge scores are supplemental judgments, not ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

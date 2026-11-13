import json
import logging
import math
import os
import re
import sqlite3
import statistics
from contextlib import closing
from contextvars import ContextVar
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / ".metrics" / "runtime.sqlite3"
_logger = logging.getLogger(__name__)
_LEXICAL_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def lexical_token_count(text: str) -> int:
    return len(_LEXICAL_TOKEN_PATTERN.findall(text))


_ACTIVE_MODEL_USAGE: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "active_model_usage", default=None
)


class ModelUsageCallback(BaseCallbackHandler):
    """Capture token counts from LangChain calls without retaining prompts."""

    def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
        usage_rows = _ACTIVE_MODEL_USAGE.get()
        if usage_rows is None:
            return
        output = response.llm_output or {}
        token_usage = output.get("token_usage") or {}
        if not token_usage:
            return
        usage_rows.append({
            "model": output.get("model_name", "unknown"),
            "input_tokens": token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)),
            "output_tokens": token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)),
        })


class ModelUsageCapture:
    def __enter__(self) -> list[dict[str, Any]]:
        self.usage: list[dict[str, Any]] = []
        self._token = _ACTIVE_MODEL_USAGE.set(self.usage)
        return self.usage

    def __exit__(
        self, _exc_type: Any, _exc_value: Any, _traceback: Any
    ) -> None:
        _ACTIVE_MODEL_USAGE.reset(self._token)


def metrics_db_path() -> Path:
    configured = os.getenv("METRICS_DB_PATH")
    return Path(configured).expanduser() if configured else _DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    path = metrics_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS query_runs (
            id INTEGER PRIMARY KEY,
            occurred_at TEXT NOT NULL,
            end_to_end_ms REAL NOT NULL,
            guardrails_ms REAL NOT NULL,
            planner_ms REAL NOT NULL,
            embedding_ms REAL NOT NULL,
            vector_search_ms REAL NOT NULL,
            retrieval_ms REAL NOT NULL,
            reranking_ms REAL NOT NULL,
            generation_ms REAL NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            estimated_cost_usd REAL,
            success INTEGER NOT NULL,
            timed_out INTEGER NOT NULL,
            failure_stage TEXT,
            retrieved_chunks INTEGER NOT NULL,
            retrieved_candidates INTEGER NOT NULL DEFAULT 0,
            model_usage_json TEXT NOT NULL,
            embedding_tokens_estimate INTEGER,
            embedding_model TEXT,
            collection_name TEXT
        )
        """
    )
    query_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(query_runs)")
    }
    if "embedding_tokens_estimate" not in query_columns:
        connection.execute(
            "ALTER TABLE query_runs ADD COLUMN embedding_tokens_estimate INTEGER"
        )
    if "retrieved_candidates" not in query_columns:
        connection.execute(
            "ALTER TABLE query_runs ADD COLUMN retrieved_candidates INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id INTEGER PRIMARY KEY,
            occurred_at TEXT NOT NULL,
            elapsed_seconds REAL NOT NULL,
            files_seen INTEGER NOT NULL,
            documents_indexed INTEGER NOT NULL,
            documents_failed INTEGER NOT NULL,
            documents_skipped INTEGER NOT NULL,
            chunks_indexed INTEGER NOT NULL,
            indexing_seconds REAL NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def record_query_run(
    *,
    end_to_end_ms: float,
    guardrails_ms: float = 0,
    planner_ms: float = 0,
    embedding_ms: float = 0,
    vector_search_ms: float = 0,
    retrieval_ms: float = 0,
    reranking_ms: float = 0,
    generation_ms: float = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float | None = None,
    success: bool,
    timed_out: bool = False,
    failure_stage: str | None = None,
    retrieved_chunks: int = 0,
    retrieved_candidates: int = 0,
    model_usage: list[dict[str, Any]] | None = None,
    embedding_tokens_estimate: int | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
) -> None:
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                """
            INSERT INTO query_runs (
                occurred_at, end_to_end_ms, guardrails_ms, planner_ms,
                embedding_ms, vector_search_ms, retrieval_ms, reranking_ms,
                generation_ms, input_tokens, output_tokens,
                estimated_cost_usd, success, timed_out, failure_stage,
                retrieved_chunks, retrieved_candidates, model_usage_json, embedding_tokens_estimate,
                embedding_model, collection_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    end_to_end_ms,
                    guardrails_ms,
                    planner_ms,
                    embedding_ms,
                    vector_search_ms,
                    retrieval_ms,
                    reranking_ms,
                    generation_ms,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                    int(success),
                    int(timed_out),
                    failure_stage,
                    retrieved_chunks,
                    retrieved_candidates,
                    json.dumps(model_usage or []),
                    embedding_tokens_estimate,
                    embedding_model,
                    collection_name,
                ),
            )


def record_ingestion_run(
    *,
    elapsed_seconds: float,
    files_seen: int,
    documents_indexed: int,
    documents_failed: int,
    documents_skipped: int,
    chunks_indexed: int,
    indexing_seconds: float = 0,
) -> None:
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                """
            INSERT INTO ingestion_runs (
                occurred_at, elapsed_seconds, files_seen, documents_indexed,
                documents_failed, documents_skipped, chunks_indexed,
                indexing_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds,
                    files_seen,
                    documents_indexed,
                    documents_failed,
                    documents_skipped,
                    chunks_indexed,
                    indexing_seconds,
                ),
            )


def estimate_query_cost_usd(
    model_usage: list[dict[str, Any]],
    embedding_model: str | None,
    embedding_tokens_estimate: int,
) -> float | None:
    """Estimate model and embedding cost from configured per-million-token rates."""
    raw_rates = os.getenv("LLM_COST_PER_MILLION_TOKENS_JSON", "").strip()
    if not raw_rates:
        return None

    try:
        rates = json.loads(raw_rates)
        total = 0.0
        for usage in model_usage:
            model_rates = rates.get(usage.get("model"))
            if not isinstance(model_rates, dict):
                return None
            input_rate = model_rates.get("input")
            output_rate = model_rates.get("output")
            if input_rate is None or output_rate is None:
                return None
            input_rate = float(input_rate)
            output_rate = float(output_rate)
            if input_rate < 0 or output_rate < 0:
                raise ValueError("Token rates cannot be negative")
            total += (
                int(usage.get("input_tokens", 0)) * input_rate
                + int(usage.get("output_tokens", 0)) * output_rate
            ) / 1_000_000
        if embedding_tokens_estimate:
            embedding_rates = rates.get(embedding_model or "")
            if not isinstance(embedding_rates, dict) or embedding_rates.get("input") is None:
                return None
            embedding_rate = float(embedding_rates["input"])
            if embedding_rate < 0:
                raise ValueError("Token rates cannot be negative")
            total += embedding_tokens_estimate * embedding_rate / 1_000_000
        return total
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        _logger.error("Invalid LLM pricing configuration; cost will be unavailable: %s", error)
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(rows: list[sqlite3.Row], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": statistics.fmean(values) if values else None,
    }


def runtime_metrics_summary() -> dict[str, Any]:
    with closing(_connect()) as connection:
        query_rows = connection.execute(
            "SELECT * FROM query_runs ORDER BY occurred_at"
        ).fetchall()
        ingestion_rows = connection.execute(
            "SELECT * FROM ingestion_runs ORDER BY occurred_at"
        ).fetchall()

    total_queries = len(query_rows)
    successes = sum(int(row["success"]) for row in query_rows)
    timeouts = sum(int(row["timed_out"]) for row in query_rows)
    costs = [
        float(row["estimated_cost_usd"])
        for row in query_rows
        if row["estimated_cost_usd"] is not None
    ]
    llm_models = sorted({
        item.get("model", "unknown")
        for row in query_rows
        for item in json.loads(row["model_usage_json"])
    })
    embedding_models = sorted({
        row["embedding_model"] for row in query_rows if row["embedding_model"]
    })
    collections = sorted({
        row["collection_name"] for row in query_rows if row["collection_name"]
    })
    indexed_docs = sum(int(row["documents_indexed"]) for row in ingestion_rows)
    failed_docs = sum(int(row["documents_failed"]) for row in ingestion_rows)
    indexed_chunks = sum(int(row["chunks_indexed"]) for row in ingestion_rows)
    ingestion_seconds = sum(float(row["elapsed_seconds"]) for row in ingestion_rows)
    indexing_seconds = sum(float(row["indexing_seconds"]) for row in ingestion_rows)

    return {
        "query_count": total_queries,
        "successful_query_rate": successes / total_queries if total_queries else None,
        "timeout_rate": timeouts / total_queries if total_queries else None,
        "failed_queries": total_queries - successes,
        "latency": {
            "end_to_end": _latency_summary(query_rows, "end_to_end_ms"),
            "guardrails": _latency_summary(query_rows, "guardrails_ms"),
            "planner": _latency_summary(query_rows, "planner_ms"),
            "retrieval": _latency_summary(query_rows, "retrieval_ms"),
            "embedding": _latency_summary(query_rows, "embedding_ms"),
            "vector_search": _latency_summary(query_rows, "vector_search_ms"),
            "reranking": _latency_summary(query_rows, "reranking_ms"),
            "generation": _latency_summary(query_rows, "generation_ms"),
        },
        "tokens": {
            "mean_input": (
                statistics.fmean(int(row["input_tokens"]) for row in query_rows)
                if total_queries
                else None
            ),
            "mean_output": (
                statistics.fmean(int(row["output_tokens"]) for row in query_rows)
                if total_queries
                else None
            ),
            "mean_embedding_query_lexical_estimate": (
                statistics.fmean(
                    int(row["embedding_tokens_estimate"])
                    for row in query_rows
                    if row["embedding_tokens_estimate"] is not None
                )
                if any(row["embedding_tokens_estimate"] is not None for row in query_rows)
                else None
            ),
        },
        "models": {
            "llm": llm_models,
            "embedding": embedding_models,
            "collections": collections,
        },
        "estimated_cost_usd": {
            "measured_queries": len(costs),
            "mean_per_measured_query": statistics.fmean(costs) if costs else None,
            "total": sum(costs) if costs else None,
        },
        "ingestion": {
            "run_count": len(ingestion_rows),
            "documents_indexed": indexed_docs,
            "documents_failed": failed_docs,
            "failed_ingestion_rate": (
                failed_docs / (indexed_docs + failed_docs)
                if indexed_docs + failed_docs
                else None
            ),
            "chunks_indexed": indexed_chunks,
            "indexing_seconds": indexing_seconds,
            "indexing_documents_per_minute": (
                indexed_docs * 60 / indexing_seconds if indexing_seconds else None
            ),
            "indexing_chunks_per_minute": (
                indexed_chunks * 60 / indexing_seconds if indexing_seconds else None
            ),
            "documents_per_minute": (
                indexed_docs * 60 / ingestion_seconds if ingestion_seconds else None
            ),
            "chunks_per_minute": (
                indexed_chunks * 60 / ingestion_seconds if ingestion_seconds else None
            ),
        },
        "storage": "local SQLite (METRICS_DB_PATH)",
        "privacy": "Request text and generated answers are not stored.",
    }


def main() -> None:
    print(json.dumps(runtime_metrics_summary(), indent=2))


if __name__ == "__main__":
    main()

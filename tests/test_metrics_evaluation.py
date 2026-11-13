import json
import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import evaluation
from app.metrics import (
    ModelUsageCallback,
    ModelUsageCapture,
    estimate_query_cost_usd,
    record_ingestion_run,
    record_query_run,
    runtime_metrics_summary,
)
from app.main import QueryRequest
from app import main as api


class RetrievalMetricTests(unittest.TestCase):
    def test_retrieval_scores_use_ranked_binary_relevance(self):
        scores = evaluation.retrieval_scores(
            ["irrelevant", "relevant", "also-relevant"],
            {"relevant", "also-relevant"},
            k=2,
        )
        self.assertEqual(scores["recall@2"], 0.5)
        self.assertEqual(scores["precision@2"], 0.5)
        self.assertEqual(scores["mrr@2"], 0.5)
        ideal_dcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(scores["ndcg@2"], (1 / math.log2(3)) / ideal_dcg)

    def test_retrieval_metrics_are_undefined_without_relevance_labels(self):
        scores = evaluation.retrieval_scores(["one", "two"], set(), k=2)
        self.assertTrue(all(value is None for value in scores.values()))

    def test_document_metrics_do_not_count_duplicate_chunks_as_extra_hits(self):
        scores = evaluation.retrieval_scores(
            ["relevant", "relevant", "irrelevant"], {"relevant"}, k=3
        )
        self.assertEqual(scores["recall@3"], 1.0)
        self.assertAlmostEqual(scores["precision@3"], 1 / 3)

    def test_quality_proxies_are_transparent_token_scores(self):
        scores = evaluation._quality_proxies(
            "API server serves Kubernetes state.",
            "API server serves Kubernetes state",
            "The API server serves Kubernetes state.",
            True,
        )
        self.assertEqual(scores["reference_token_f1"], 1.0)
        self.assertEqual(scores["context_lexical_support_proxy"], 1.0)


class RuntimeMetricsTests(unittest.TestCase):
    def test_query_and_ingestion_metrics_are_persisted_and_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "runtime.sqlite3")
            with patch.dict(os.environ, {"METRICS_DB_PATH": db_path}, clear=False):
                record_query_run(
                    end_to_end_ms=100,
                    retrieval_ms=20,
                    generation_ms=50,
                    input_tokens=10,
                    output_tokens=5,
                    estimated_cost_usd=0.01,
                    success=True,
                )
                record_query_run(
                    end_to_end_ms=250,
                    retrieval_ms=60,
                    generation_ms=150,
                    input_tokens=20,
                    output_tokens=10,
                    success=False,
                    timed_out=True,
                    failure_stage="generation",
                )
                record_ingestion_run(
                    elapsed_seconds=30,
                    files_seen=3,
                    documents_indexed=2,
                    documents_failed=1,
                    documents_skipped=0,
                    chunks_indexed=10,
                    indexing_seconds=20,
                )

                summary = runtime_metrics_summary()

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["successful_query_rate"], 0.5)
        self.assertEqual(summary["timeout_rate"], 0.5)
        self.assertEqual(summary["latency"]["end_to_end"]["p50_ms"], 100)
        self.assertEqual(summary["latency"]["end_to_end"]["p95_ms"], 250)
        self.assertEqual(summary["tokens"]["mean_input"], 15)
        self.assertEqual(summary["estimated_cost_usd"]["measured_queries"], 1)
        self.assertEqual(summary["ingestion"]["documents_failed"], 1)
        self.assertEqual(summary["ingestion"]["chunks_indexed"], 10)

    def test_existing_database_migrates_embedding_and_candidate_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "runtime.sqlite3")
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE query_runs (
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
                    model_usage_json TEXT NOT NULL,
                    embedding_model TEXT,
                    collection_name TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO query_runs VALUES (
                    1, 'existing', 10, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, NULL, 1, 0, NULL, 2, '[]', NULL, NULL
                )
                """
            )
            connection.commit()
            connection.close()

            with patch.dict(os.environ, {"METRICS_DB_PATH": db_path}, clear=False):
                record_query_run(
                    end_to_end_ms=25,
                    retrieved_candidates=15,
                    embedding_tokens_estimate=4,
                    success=True,
                )
                summary = runtime_metrics_summary()

            connection = sqlite3.connect(db_path)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(query_runs)")
            }
            row_count = connection.execute("SELECT COUNT(*) FROM query_runs").fetchone()[0]
            connection.close()

        self.assertIn("embedding_tokens_estimate", columns)
        self.assertIn("retrieved_candidates", columns)
        self.assertEqual(row_count, 2)
        self.assertEqual(summary["tokens"]["mean_embedding_query_lexical_estimate"], 4)

    def test_usage_callback_records_tokens_without_prompt_contents(self):
        with ModelUsageCapture() as usage:
            ModelUsageCallback().on_llm_end(
                SimpleNamespace(
                    llm_output={
                        "model_name": "example-model",
                        "token_usage": {"prompt_tokens": 23, "completion_tokens": 7},
                    }
                )
            )
        self.assertEqual(
            usage,
            [{"model": "example-model", "input_tokens": 23, "output_tokens": 7}],
        )

    def test_cost_requires_configured_rates_for_every_component(self):
        rates = json.dumps({
            "example-llm": {"input": 1.0, "output": 2.0},
            "example-embedding": {"input": 0.5},
        })
        with patch.dict(os.environ, {"LLM_COST_PER_MILLION_TOKENS_JSON": rates}):
            cost = estimate_query_cost_usd(
                [{"model": "example-llm", "input_tokens": 1000, "output_tokens": 500}],
                "example-embedding",
                200,
            )
        self.assertAlmostEqual(cost, 0.0021)

    def test_api_records_successful_query_metrics_without_query_text(self):
        state = {
            "final_answer": "answer",
            "plan": [],
            "status": "Response generated.",
            "documents": ["context"],
            "retrieved_chunk_ids": ["chunk-1"],
            "retrieved_candidates": 15,
            "planner_ms": 12,
            "retrieval_ms": 30,
            "model_usage": [{
                "model": "answer-model",
                "input_tokens": 40,
                "output_tokens": 10,
            }],
        }
        guard_usage = [{
            "model": "guard-model",
            "input_tokens": 8,
            "output_tokens": 2,
        }]
        with (
            patch.object(api, "guard", return_value=(False, None, guard_usage)),
            patch.object(api.rag_agent, "invoke", return_value=state),
            patch.object(api, "record_query_run") as record,
            patch.object(api, "estimate_query_cost_usd", return_value=0.01),
        ):
            result = api.query(QueryRequest(q="do not persist this text"))

        self.assertEqual(result["answer"], "answer")
        metrics = record.call_args.kwargs
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["retrieved_chunks"], 1)
        self.assertEqual(metrics["retrieved_candidates"], 15)
        self.assertEqual(metrics["input_tokens"], 48)
        self.assertEqual(metrics["output_tokens"], 12)
        self.assertNotIn("do not persist this text", str(metrics))

    def test_api_records_graph_failures_and_surfaces_http_error(self):
        from fastapi import HTTPException

        with (
            patch.object(api, "guard", return_value=(False, None, [])),
            patch.object(api.rag_agent, "invoke", side_effect=RuntimeError("provider failed")),
            patch.object(api, "record_query_run") as record,
        ):
            with self.assertRaises(HTTPException) as raised:
                api.query(QueryRequest(q="question"))

        self.assertEqual(raised.exception.status_code, 502)
        self.assertFalse(record.call_args.kwargs["success"])
        self.assertEqual(record.call_args.kwargs["failure_stage"], "agent_graph")


class EvaluationReportTests(unittest.TestCase):
    def test_report_contains_reproducibility_and_both_retrieval_modes(self):
        class FakeQdrant:
            def __init__(self, **_kwargs):
                pass

            def get_collection(self, _name):
                return SimpleNamespace(
                    points_count=2,
                    status="green",
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors=SimpleNamespace(size=3, distance="COSINE")
                        )
                    ),
                )

            def query_points(self, **_kwargs):
                return SimpleNamespace(points=[
                    SimpleNamespace(
                        id="chunk-1",
                        payload={
                            "source": "doc.txt",
                            "source_type": "true",
                            "chunk_index": 0,
                            "text": "The API server serves requests.",
                        },
                    ),
                    SimpleNamespace(
                        id="chunk-2",
                        payload={
                            "source": "other.txt",
                            "source_type": "true",
                            "chunk_index": 0,
                            "text": "Unrelated content.",
                        },
                    ),
                ])

            def scroll(self, **kwargs):
                points = [
                    SimpleNamespace(
                        id="chunk-1",
                        payload={"source": "doc.txt", "source_type": "true", "chunk_index": 0, "text": "The API server serves requests."},
                    ),
                    SimpleNamespace(
                        id="chunk-2",
                        payload={"source": "other.txt", "source_type": "true", "chunk_index": 0, "text": "Unrelated content."},
                    ),
                ]
                return points, None

        dataset = {
            "name": "test-eval",
            "version": "test-v1",
            "examples": [{
                "id": "test-1",
                "question": "What does the API server do?",
                "expected_answer": "The API server serves requests.",
                "answerable": True,
                "relevant_document_ids": ["doc.txt"],
                "relevant_chunk_ids": ["doc.txt#0"],
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "doc.json").write_text(json.dumps({
                "filename": "doc.txt",
                "chunks": ["The API server serves requests."],
            }), encoding="utf-8")
            (corpus / "other.json").write_text(json.dumps({
                "filename": "other.txt",
                "chunks": ["Unrelated content."],
            }), encoding="utf-8")
            report_path = root / "report.json"

            with (
                patch.object(evaluation, "QdrantClient", FakeQdrant),
                patch.object(evaluation, "embed_query", return_value=[0.1]),
                patch.object(evaluation, "get_embedding_model_name", return_value="embedding-v1"),
                patch.object(
                    evaluation,
                    "rerank_documents_with_indices",
                    return_value=[(0, "The API server serves requests.")],
                ),
                patch.object(evaluation, "runtime_metrics_summary", return_value={"query_count": 0}),
                patch.dict(
                    "sys.modules",
                    {"app.agents.nodes.responder": SimpleNamespace(
                        generate_node=lambda _state: {"final_answer": "The API server serves requests."}
                    )},
                ),
            ):
                report = evaluation.run_evaluation(
                    dataset_path,
                    report_path,
                    k=1,
                    candidate_k=2,
                    processed_data=corpus,
                )

            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["dataset"]["version"], "test-v1")
        self.assertEqual(report["reproducibility"]["models"]["embedding"], "embedding-v1")
        self.assertEqual(
            report["retrieval"]["semantic_plus_flashrank_document_metrics"]["recall@1"],
            1.0,
        )
        self.assertIn("semantic_only", report["examples"][0]["retrieval"])
        self.assertEqual(saved["answer_quality"]["human_reviewed_examples"], 0)


if __name__ == "__main__":
    unittest.main()

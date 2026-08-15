import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.rag import retrieval
from src.rag.retrieval import RetrievalResult


class RetrievalParallelTests(unittest.TestCase):
    def test_retrieval_max_workers_bounds_query_count(self):
        with patch.dict("os.environ", {"RAG_RETRIEVAL_MAX_WORKERS": "8"}, clear=False):
            self.assertEqual(retrieval.retrieval_max_workers(3), 3)

    def test_retrieval_max_workers_uses_parallel_workers_with_rerank(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(retrieval.retrieval_max_workers(3, rerank=True), 3)

    def test_multi_query_hybrid_retrieve_merges_parallel_results(self):
        def fake_hybrid_retrieve(query: str, **kwargs):
            return [
                RetrievalResult(
                    id=f"id-{query}",
                    document=f"Evidence for {query}",
                    metadata={"url": f"https://example.com/{query}"},
                    score=1.0,
                    semantic_score=1.0,
                    bm25_score=0.0,
                )
            ]

        with patch.dict("os.environ", {"RAG_RETRIEVAL_MAX_WORKERS": "2"}, clear=False):
            with patch("src.rag.retrieval.hybrid_retrieve", side_effect=fake_hybrid_retrieve):
                results = retrieval.multi_query_hybrid_retrieve(
                    ["alpha", "beta"],
                    top_k=2,
                    per_query_k=1,
                    diversify_urls=False,
                )

        self.assertEqual({result.id for result in results}, {"id-alpha", "id-beta"})

    def test_multi_query_hybrid_retrieve_reranks_merged_candidates_once(self):
        hybrid_calls = []

        def fake_hybrid_retrieve(query: str, **kwargs):
            hybrid_calls.append((query, kwargs.get("rerank")))
            return [
                RetrievalResult(
                    id=f"id-{query}",
                    document=f"Evidence for {query}",
                    metadata={"url": f"https://example.com/{query}"},
                    score=1.0,
                    semantic_score=1.0,
                    bm25_score=0.0,
                )
            ]

        def fake_rerank_results(query: str, results, **kwargs):
            self.assertIn("alpha", query)
            self.assertIn("beta", query)
            self.assertEqual({result.id for result in results}, {"id-alpha", "id-beta"})
            return list(results)

        with patch.dict("os.environ", {"RAG_RETRIEVAL_MAX_WORKERS": "2"}, clear=False):
            with patch("src.rag.retrieval.hybrid_retrieve", side_effect=fake_hybrid_retrieve):
                with patch("src.rag.retrieval.rerank_results", side_effect=fake_rerank_results) as rerank_mock:
                    results = retrieval.multi_query_hybrid_retrieve(
                        ["alpha", "beta"],
                        top_k=2,
                        per_query_k=1,
                        diversify_urls=False,
                        rerank=True,
                    )

        self.assertEqual({result.id for result in results}, {"id-alpha", "id-beta"})
        self.assertEqual([rerank for _, rerank in hybrid_calls], [False, False])
        self.assertEqual(rerank_mock.call_count, 1)

    def test_rerank_results_falls_back_when_reranker_fails(self):
        results = [
            RetrievalResult(
                id="a",
                document="Alpha evidence",
                metadata={"url": "https://example.com/a"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="b",
                document="Beta evidence",
                metadata={"url": "https://example.com/b"},
                score=0.8,
                semantic_score=0.8,
                bm25_score=0.0,
            ),
        ]
        buffer = StringIO()

        with patch("src.rag.retrieval.langchain_cross_encoder_reranker", side_effect=NotImplementedError("Cannot copy out of meta tensor; no data!")):
            with redirect_stdout(buffer):
                reranked = retrieval.rerank_results("query", results, top_n=2)

        self.assertEqual([result.id for result in reranked], ["a", "b"])
        self.assertIn("Cannot copy out of meta tensor; no data!", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

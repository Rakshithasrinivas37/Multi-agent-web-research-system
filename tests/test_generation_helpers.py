import unittest

from src.rag.generation import audit_synthesis_citations, precision_retrieval_queries, report_supporting_chunks
from src.rag.retrieval import RetrievalResult


class GenerationHelperTests(unittest.TestCase):
    def test_audit_synthesis_citations_flags_invalid_markers(self):
        sources = [{"index": 1}, {"index": 2}, {"index": 4}]

        audit = audit_synthesis_citations("Uses valid [1], converted 【2】, and invalid [9].", sources)

        self.assertEqual(audit["referenced_source_indexes"], [1, 2, 9])
        self.assertEqual(audit["valid_referenced_source_indexes"], [1, 2])
        self.assertEqual(audit["invalid_source_indexes"], [9])
        self.assertEqual(audit["uncited_source_indexes"], [4])
        self.assertTrue(audit["has_invalid_citations"])

    def test_report_supporting_chunks_prioritizes_cited_sources(self):
        results = [
            RetrievalResult(
                id="secondary",
                document="Source: Secondary\nURL: https://example.com/secondary\nTask: task\n"
                + "Secondary evidence text with enough content for meaningful evidence. " * 3,
                metadata={
                    "title": "Secondary",
                    "url": "https://example.com/secondary",
                    "source_type": "article",
                },
                score=0.95,
                semantic_score=0.95,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="primary",
                document="Source: Primary\nURL: https://arxiv.org/abs/1234.5678\nTask: task\n"
                + "Primary evidence text with enough content for meaningful evidence. " * 3,
                metadata={
                    "title": "Primary",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "source_type": "arxiv",
                },
                score=0.10,
                semantic_score=0.10,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "secondary"},
            {"index": 2, "id": "primary"},
        ]

        chunks = report_supporting_chunks(results, sources, max_chunks=1, cited_source_indexes=[1])

        self.assertEqual(chunks[0]["source_index"], 1)

    def test_precision_retrieval_queries_adds_exact_evidence_terms(self):
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": [
                "What are the equations of additive attention?",
                "What benchmark results demonstrate GLUE and ImageNet performance?",
                "Where are the official TensorFlow and PyTorch API references?",
            ],
            "tasks": [
                {
                    "query_context": "What are the equations of additive attention?",
                    "extraction_goal": "Find the original paper formula.",
                    "expected_signals": ["equation", "formula"],
                }
            ],
        }

        queries = precision_retrieval_queries(plan)
        joined = " ".join(queries).lower()

        self.assertIn("exact equation formula", joined)
        self.assertIn("benchmark table results", joined)
        self.assertIn("official documentation api signature", joined)


if __name__ == "__main__":
    unittest.main()

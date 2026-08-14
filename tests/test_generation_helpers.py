import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.rag.generation import (
    audit_synthesis_citations,
    browser_signal_results,
    high_signal_browser_snippets,
    is_valid_retrieval_query,
    llm_sub_question_retrieval_query_result,
    parse_llm_retrieval_queries,
    precision_retrieval_queries,
    print_synthesis_chunks,
    planner_tasks_to_rag_queries,
    report_supporting_chunks,
    select_synthesis_context,
    sub_question_retrieval_queries,
    valid_retrieval_queries,
)
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

    def test_sub_question_retrieval_queries_expand_each_question_generically(self):
        queries = sub_question_retrieval_queries(
            "How is a model deployed and monitored in production?",
            objective="MLOps system",
            task_details="Find lifecycle evidence.",
        )

        self.assertEqual(len(queries), 3)
        self.assertIn("Find lifecycle evidence", queries[0])
        self.assertIn("overview background concepts methods evidence", queries[1])
        self.assertIn("source-backed details examples definitions", queries[2])

    def test_planner_tasks_to_rag_queries_splits_sub_questions_into_variants(self):
        plan = {
            "objective": "Model deployment",
            "sub_questions": [
                "How is a model deployed?",
                "What monitoring metrics are used?",
            ],
        }

        queries = planner_tasks_to_rag_queries(plan)

        self.assertGreaterEqual(len(queries), 5)
        self.assertTrue(any("source-backed details" in query for query in queries))
        self.assertIn("Model deployment", queries[-1])

    def test_parse_llm_retrieval_queries_reads_json_queries_only(self):
        raw = """```json
{"items":[{"sub_question":"What is attention?","queries":["attention definition equation","attention examples"]}]}
```"""

        queries = parse_llm_retrieval_queries(raw)

        self.assertEqual(queries, ["attention definition equation", "attention examples"])

    def test_parse_llm_retrieval_queries_recovers_json_like_query_arrays(self):
        raw = (
            '{"items":[{"sub_question":"What is attention?",'
            '"queries":["attention definition equation","attention benchmark evidence",]}]}'
        )

        queries = parse_llm_retrieval_queries(raw)

        self.assertEqual(queries, ["attention definition equation", "attention benchmark evidence"])

    def test_valid_retrieval_queries_filters_placeholders(self):
        plan = {"objective": "Attention mechanism", "sub_questions": ["What is attention?"]}

        queries = valid_retrieval_queries(["query 1", "query 2", "attention mechanism definition equation"], plan)

        self.assertEqual(queries, ["attention mechanism definition equation"])

    def test_is_valid_retrieval_query_rejects_generic_outputs(self):
        self.assertFalse(is_valid_retrieval_query("query 1", {"attention"}))
        self.assertFalse(is_valid_retrieval_query("example query", {"attention"}))
        self.assertTrue(is_valid_retrieval_query("attention mechanism definition equation", {"attention"}))
        self.assertTrue(is_valid_retrieval_query("official implementation signature batched projections", {"attention"}))

    @patch.dict("os.environ", {}, clear=True)
    def test_llm_sub_question_retrieval_query_result_defaults_to_qwen(self):
        result = llm_sub_question_retrieval_query_result(
            {"objective": "X research", "sub_questions": ["What is X?"]},
        )

        self.assertEqual(result["model"], "qwen/qwen3.6-27b")
        self.assertEqual(result["error"], "GROQ_API_KEY is not set")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "RAG_SUBQUESTION_QUERY_MODEL": "llama-test"}, clear=False)
    @patch("src.rag.generation.create_chat_completion_with_retries")
    def test_llm_sub_question_retrieval_query_result_uses_query_model(self, completion):
        class Message:
            content = '{"items":[{"sub_question":"What is X?","queries":["x research overview","x research evidence details"]}]}'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            model = "llama-test"

        completion.return_value = Response()

        result = llm_sub_question_retrieval_query_result(
            {"objective": "X research", "sub_questions": ["What is X?"]},
        )

        self.assertEqual(result["queries"], ["x research overview", "x research evidence details"])
        self.assertEqual(result["model"], "llama-test")
        self.assertEqual(result["error"], "")
        self.assertEqual(completion.call_args.kwargs["model"], "llama-test")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "RAG_SUBQUESTION_QUERY_MODEL": "llama-test"}, clear=False)
    @patch("src.rag.generation.create_chat_completion_with_retries")
    def test_llm_sub_question_retrieval_query_result_rejects_placeholder_queries(self, completion):
        class Message:
            content = '{"items":[{"sub_question":"What is X?","queries":["query 1","query 2"]}]}'

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            model = "llama-test"

        completion.return_value = Response()

        result = llm_sub_question_retrieval_query_result(
            {"objective": "X research", "sub_questions": ["What is X?"]},
        )

        self.assertEqual(result["queries"], [])
        self.assertEqual(result["error"], "LLM query rewrite returned no usable queries")

    def test_browser_signal_results_promotes_formula_evidence(self):
        browser_results = [
            {
                "sources": [
                    {
                        "title": "Primary Paper",
                        "url": "https://arxiv.org/pdf/1234.56789",
                        "full_content": (
                            "Background text. The equation is y = softmax(x) and the benchmark score "
                            "is 92.1 accuracy on the held-out test set. More surrounding explanation."
                        ),
                    }
                ]
            }
        ]

        results = browser_signal_results(browser_results, objective="technical report")

        self.assertEqual(len(results), 1)
        self.assertIn("equation", results[0].document)
        self.assertEqual(results[0].metadata["url"], "https://arxiv.org/pdf/1234.56789")
        self.assertEqual(results[0].metadata["source_quality"], "high_signal_browser")

    def test_high_signal_browser_snippets_prioritizes_formula_dense_text(self):
        content = (
            "The benchmark score is 91.2 accuracy on a public dataset. "
            + "General background text without exact implementation details. " * 20
            + "The paper gives the exact formula y = softmax(Wx + b), where W is learned."
        )

        snippets = high_signal_browser_snippets(content, max_snippets=1)

        self.assertEqual(len(snippets), 1)
        self.assertIn("y = softmax", snippets[0])

    def test_print_synthesis_chunks_shows_primary_source_after_reranking(self):
        result = RetrievalResult(
            id="paper",
            document="Attention evidence from the original paper with enough preview text.",
            metadata={"title": "Paper", "url": "https://arxiv.org/pdf/1706.03762"},
            score=0.87,
            semantic_score=0.5,
            bm25_score=0.4,
            rerank_score=1.0,
        )
        buffer = StringIO()

        with patch.dict("os.environ", {"RAG_PRINT_SYNTHESIS_CHUNKS_LIMIT": "1"}, clear=False):
            with redirect_stdout(buffer):
                print_synthesis_chunks([result], label="test")

        output = buffer.getvalue()
        self.assertIn("chunks passed to synthesizer after reranking (test): 1", output)
        self.assertIn("primary/paper", output)
        self.assertIn("https://arxiv.org/pdf/1706.03762", output)
        self.assertIn("rerank=1.000", output)

    def test_select_synthesis_context_balances_planner_questions(self):
        results = [
            RetrievalResult(
                id=f"general-{index}",
                document="Attention computes weighted context vectors in neural networks. " * 4,
                metadata={"title": "General attention", "url": "https://example.com/general"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
            for index in range(8)
        ]
        results.extend(
            [
                RetrievalResult(
                    id="pytorch",
                    document="torch.nn.MultiheadAttention is the official PyTorch API for multi-head attention. " * 3,
                    metadata={"title": "PyTorch MultiheadAttention", "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"},
                    score=0.8,
                    semantic_score=0.8,
                    bm25_score=0.0,
                ),
                RetrievalResult(
                    id="vit",
                    document="Vision Transformer applies self-attention to image patches for image classification. " * 3,
                    metadata={"title": "Vision Transformer", "url": "https://arxiv.org/pdf/2010.11929"},
                    score=0.7,
                    semantic_score=0.7,
                    bm25_score=0.0,
                ),
            ]
        )

        selected = select_synthesis_context(
            results,
            [
                "What is the official PyTorch API for multi-head attention and how is it used?",
                "How are attention mechanisms applied in Vision Transformers for image data?",
            ],
            max_chunks=4,
            per_question=1,
        )
        selected_ids = {result.id for result in selected}

        self.assertIn("pytorch", selected_ids)
        self.assertIn("vit", selected_ids)
        self.assertLessEqual(len(selected), 4)

    def test_select_synthesis_context_filters_weak_chunks(self):
        selected = select_synthesis_context(
            [
                RetrievalResult(
                    id="empty",
                    document="",
                    metadata={"title": "Empty", "url": "https://example.com/empty"},
                    score=1.0,
                    semantic_score=1.0,
                    bm25_score=0.0,
                ),
                RetrievalResult(
                    id="useful",
                    document="Performer approximates softmax attention with random feature maps for efficient attention. " * 3,
                    metadata={"title": "Performer", "url": "https://arxiv.org/pdf/2009.14794"},
                    score=0.8,
                    semantic_score=0.8,
                    bm25_score=0.0,
                ),
            ],
            ["What are efficient attention methods such as Performer?"],
        )

        self.assertEqual([result.id for result in selected], ["useful"])


if __name__ == "__main__":
    unittest.main()

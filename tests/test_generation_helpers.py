import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.rag.generation import (
    audit_synthesis_citations,
    broad_query_hints,
    build_sub_question_evidence_packs,
    build_coverage_by_question,
    build_generation_context,
    browser_signal_results,
    compact_retrieved_chunks,
    complete_sub_question_query_coverage,
    coverage_gap_items,
    evidence_focused_question_query,
    fallback_gap_retrieval_queries,
    high_signal_browser_snippets,
    is_valid_retrieval_query,
    llm_sub_question_retrieval_query_result,
    parse_gap_query_lines,
    parse_llm_retrieval_queries,
    planner_question_source_urls,
    planner_sub_question_specs,
    precision_retrieval_queries,
    print_synthesis_chunks,
    planner_tasks_to_rag_queries,
    report_supporting_chunks,
    retrieve_full_collection_enabled,
    retrieval_topic_phrase,
    result_supports_question,
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

    def test_build_sub_question_evidence_packs_groups_matching_chunks(self):
        results = [
            RetrievalResult(
                id="api",
                document="Official API reference explains implementation parameters and usage examples. " * 3,
                metadata={"title": "API docs", "url": "https://docs.example.com/api"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="benchmark",
                document="Benchmark results report accuracy, latency, and evaluation metrics. " * 3,
                metadata={"title": "Benchmark report", "url": "https://example.com/benchmarks"},
                score=0.8,
                semantic_score=0.8,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "api", "title": "API docs", "url": "https://docs.example.com/api"},
            {"index": 2, "id": "benchmark", "title": "Benchmark report", "url": "https://example.com/benchmarks"},
        ]

        packs = build_sub_question_evidence_packs(
            ["What is the implementation API?", "What benchmark result is reported?"],
            results,
            sources,
        )

        self.assertEqual(packs[0]["chunks"][0]["source_index"], 1)
        self.assertEqual(packs[1]["chunks"][0]["source_index"], 2)

    def test_build_sub_question_evidence_packs_prefers_planned_source_url(self):
        results = [
            RetrievalResult(
                id="secondary",
                document="API API API general tutorial text with enough content for meaningful evidence. " * 3,
                metadata={"title": "Tutorial", "url": "https://example.com/tutorial"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="official",
                document="Official reference documents constructor arguments and usage examples. " * 3,
                metadata={"title": "Official docs", "url": "https://docs.example.com/api"},
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "secondary", "url": "https://example.com/tutorial"},
            {"index": 2, "id": "official", "url": "https://docs.example.com/api"},
        ]

        packs = build_sub_question_evidence_packs(
            ["What is the API usage?"],
            results,
            sources,
            question_source_urls={"What is the API usage?": ["https://docs.example.com/api"]},
            max_chunks_per_question=1,
        )

        self.assertEqual(packs[0]["chunks"][0]["source_index"], 2)

    def test_evidence_pack_does_not_mark_wrong_evidence_type_covered(self):
        results = [
            RetrievalResult(
                id="luong-benchmark",
                document="Luong attention reports WMT BLEU gains and benchmark results from translation experiments. " * 3,
                metadata={"title": "Luong paper", "url": "https://arxiv.org/pdf/1508.04025", "source_type": "arxiv"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "luong-benchmark", "url": "https://arxiv.org/pdf/1508.04025"}]

        packs = build_sub_question_evidence_packs(
            ["How does multiplicative attention work, including its equations?"],
            results,
            sources,
            question_source_urls={
                "How does multiplicative attention work, including its equations?": ["https://arxiv.org/pdf/1508.04025"]
            },
        )

        self.assertEqual(packs[0]["coverage"], "partial")
        self.assertEqual(packs[0]["chunks"][0]["source_index"], 1)

    def test_planner_question_source_urls_maps_task_context_ids(self):
        plan = {
            "sub_questions": ["What is the core equation?"],
            "sub_question_specs": [{"question_id": "q001", "question": "What is the core equation?"}],
            "tasks": [{"query_context": "q001", "url": "https://arxiv.org/abs/1234.56789"}],
        }

        mapping = planner_question_source_urls(plan)

        self.assertEqual(mapping["What is the core equation?"], ["https://arxiv.org/pdf/1234.56789"])

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

    def test_planner_sub_question_specs_uses_structured_specs_when_available(self):
        plan = {
            "sub_questions": ["What is the core equation?"],
            "sub_question_specs": [
                {
                    "question_id": "q001",
                    "question": "What is the core equation?",
                    "required_evidence": ["equation"],
                }
            ],
        }

        specs = planner_sub_question_specs(plan)

        self.assertEqual(specs[0]["question_id"], "q001")
        self.assertEqual(specs[0]["required_evidence"], ["equation"])

    def test_planner_sub_question_specs_falls_back_to_plain_questions(self):
        specs = planner_sub_question_specs({"sub_questions": ["What benchmark results show performance?"]})

        self.assertEqual(specs[0]["question_id"], "q001")
        self.assertIn("benchmark", specs[0]["required_evidence"])

    def test_build_coverage_by_question_maps_status_and_citations(self):
        specs = [
            {"question_id": "q001", "question": "What is the core equation?", "required_evidence": ["equation"]},
            {"question_id": "q002", "question": "What benchmark results show performance?", "required_evidence": ["benchmark"]},
        ]
        synthesis = """
What is the core equation?
Covered with equation evidence [1].

What benchmark results show performance?
Missing Evidence: exact benchmark values are not present.
"""

        context = [
            RetrievalResult(
                id="equation",
                document="The core equation is y = softmax(QK^T / sqrt(d_k)) V with enough surrounding evidence text.",
                metadata={"title": "Equation source", "url": "https://arxiv.org/abs/1234.5678"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        coverage = build_coverage_by_question(synthesis, specs, [{"index": 1, "id": "equation"}], retrieved_context=context)

        self.assertEqual(coverage[0]["status"], "covered")
        self.assertEqual(coverage[0]["source_indexes"], [1])
        self.assertTrue(coverage[0]["has_citations"])
        self.assertEqual(coverage[1]["status"], "missing")
        self.assertFalse(coverage[1]["has_citations"])
        self.assertTrue(coverage[1]["missing_reason"])

    def test_build_coverage_by_question_uses_retrieved_context_sidecar(self):
        specs = [
            {"question_id": "q001", "question": "What is the implementation API?", "required_evidence": ["api"]},
            {"question_id": "q002", "question": "What benchmark result is reported?", "required_evidence": ["benchmark"]},
        ]
        sources = [{"index": 1, "id": "api-chunk"}]
        context = [
            RetrievalResult(
                id="api-chunk",
                document="The implementation API exposes an official function with parameters and usage examples.",
                metadata={"title": "API docs", "url": "https://docs.example.com/api"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        coverage = build_coverage_by_question(
            "The synthesis explains the topic naturally with source support [1].",
            specs,
            sources,
            retrieved_context=context,
        )

        self.assertEqual(coverage[0]["status"], "covered")
        self.assertEqual(coverage[0]["source_indexes"], [1])
        self.assertEqual(coverage[0]["evidence_count"], 1)
        self.assertEqual(coverage[1]["status"], "missing")
        self.assertIn("benchmark", coverage_gap_items(coverage)[0].lower())

    def test_build_coverage_requires_topic_match_for_strict_evidence(self):
        specs = [
            {
                "question_id": "q001",
                "question": "What are the known limitations of attention mechanisms?",
                "required_evidence": ["limitations"],
            }
        ]
        context = [
            RetrievalResult(
                id="generic-limitations",
                document="Limitations in research are constraints that affect what can be concluded from a study. " * 3,
                metadata={"title": "Limitations in Research", "url": "https://example.com/limitations"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        coverage = build_coverage_by_question(
            "Known limitations of attention mechanisms are discussed [1].",
            specs,
            [{"index": 1, "id": "generic-limitations"}],
            retrieved_context=context,
        )

        self.assertEqual(coverage[0]["status"], "partial")
        self.assertEqual(coverage[0]["evidence_count"], 0)

    def test_build_generation_context_dedupes_sources_by_url(self):
        results = [
            RetrievalResult(
                id="chunk-a",
                document="First source chunk with enough evidence about the topic. " * 4,
                metadata={"title": "Shared Source", "url": "https://example.com/source"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="chunk-b",
                document="Second source chunk with more evidence from the same source URL. " * 4,
                metadata={"title": "Shared Source", "url": "https://example.com/source"},
                score=0.8,
                semantic_score=0.8,
                bm25_score=0.0,
            ),
        ]

        context, sources = build_generation_context(results)
        chunks = compact_retrieved_chunks(results, sources)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["ids"], ["chunk-a", "chunk-b"])
        self.assertEqual(context.count("[1] Shared Source"), 2)
        self.assertEqual({chunk["source_index"] for chunk in chunks}, {1})

    def test_compact_retrieved_chunks_keeps_full_content_when_uncapped(self):
        tail = "critical evidence at the end"
        results = [
            RetrievalResult(
                id="chunk-long",
                document=("Long retrieved context with useful evidence. " * 120) + tail,
                metadata={"title": "Long Source", "url": "https://example.com/long"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "url": "https://example.com/long", "ids": ["chunk-long"]}]

        chunks = compact_retrieved_chunks(results, sources, max_chars=None)

        self.assertIn(tail, chunks[0]["content"])

    def test_build_coverage_by_question_uses_evidence_pack_when_synthesis_section_is_generic(self):
        specs = [{"question_id": "q001", "question": "What is the official API?", "required_evidence": ["api"]}]
        packs = [
            {
                "question": "What is the official API?",
                "coverage": "covered",
                "chunks": [{"source_index": 2, "content": "Official API usage"}],
            }
        ]

        coverage = build_coverage_by_question("API details are available in official docs.", specs, [{"index": 2}], evidence_packs=packs)

        self.assertEqual(coverage[0]["status"], "covered")
        self.assertEqual(coverage[0]["source_indexes"], [2])
        self.assertEqual(coverage[0]["evidence_count"], 1)

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

    def test_parse_llm_retrieval_queries_strips_thinking_blocks(self):
        raw = (
            "<think>Here is my reasoning. It should not become a query.</think>\n"
            '{"items":[{"sub_question":"What is attention?",'
            '"queries":["attention definition equation","attention api implementation"]}]}'
        )

        queries = parse_llm_retrieval_queries(raw)

        self.assertEqual(queries, ["attention definition equation", "attention api implementation"])

    def test_valid_retrieval_queries_filters_placeholders(self):
        plan = {"objective": "Attention mechanism", "sub_questions": ["What is attention?"]}

        queries = valid_retrieval_queries(["query 1", "query 2", "attention mechanism definition equation"], plan)

        self.assertEqual(queries, ["attention mechanism definition equation"])

    def test_valid_retrieval_queries_filters_reasoning_output(self):
        plan = {"objective": "Attention mechanism", "sub_questions": ["What is attention?"]}

        queries = valid_retrieval_queries(
            [
                "<think>Here's a thinking process: analyze user input and return JSON only.</think>",
                "attention mechanism scaled dot product equation",
            ],
            plan,
        )

        self.assertEqual(queries, ["attention mechanism scaled dot product equation"])

    def test_is_valid_retrieval_query_rejects_generic_outputs(self):
        self.assertFalse(is_valid_retrieval_query("query 1", {"attention"}))
        self.assertFalse(is_valid_retrieval_query("example query", {"attention"}))
        self.assertTrue(is_valid_retrieval_query("attention mechanism definition equation", {"attention"}))
        self.assertTrue(is_valid_retrieval_query("official implementation signature batched projections", {"attention"}))

    def test_sub_question_retrieval_queries_are_broad(self):
        queries = sub_question_retrieval_queries(
            "What benchmark results compare model performance?",
            objective="Model architecture",
        )
        joined = " ".join(queries).lower()

        self.assertIn("overview", joined)
        self.assertIn("benchmark", joined)
        self.assertIn("metrics", joined)
        self.assertTrue(any(query.lower().startswith(("what ", "which ", "where ")) for query in queries))
        self.assertNotIn("source-backed details examples definitions equations metrics implementation limitations", joined)

    def test_retrieval_topic_phrase_removes_question_filler(self):
        phrase = retrieval_topic_phrase(
            "What recent efficient-attention architectures (e.g., Linformer, Performer, Longformer) "
            "propose and how do they trade off performance vs. cost?"
        )

        self.assertIn("Linformer", phrase)
        self.assertIn("Performer", phrase)
        self.assertIn("Longformer", phrase)
        self.assertNotIn("What", phrase)
        self.assertNotIn("they", phrase.lower().split())

    def test_retrieve_full_collection_enabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(retrieve_full_collection_enabled())
        with patch.dict("os.environ", {"RAG_RETRIEVE_FULL_COLLECTION": "false"}):
            self.assertFalse(retrieve_full_collection_enabled())

    def test_result_supports_question_requires_requested_evidence_type(self):
        question = "How does multiplicative attention work, including its equations?"
        benchmark_only = RetrievalResult(
            id="benchmark",
            document="Luong attention improves WMT BLEU benchmark results from translation experiments. " * 2,
            metadata={"title": "Luong paper", "url": "https://arxiv.org/pdf/1508.04025", "source_type": "arxiv"},
            score=1.0,
            semantic_score=1.0,
            bm25_score=0.0,
        )
        equation_chunk = RetrievalResult(
            id="equation",
            document="Multiplicative attention uses a score function with a dot product equation h_t^T W h_s = value.",
            metadata={"title": "Luong paper", "url": "https://arxiv.org/pdf/1508.04025", "source_type": "arxiv"},
            score=1.0,
            semantic_score=1.0,
            bm25_score=0.0,
        )

        self.assertFalse(result_supports_question(question, benchmark_only))
        self.assertTrue(result_supports_question(question, equation_chunk))

    def test_evidence_focused_question_query_adds_evidence_hints(self):
        query = evidence_focused_question_query("How does multiplicative attention work, including its equations?")

        self.assertIn("equation", query.lower())
        self.assertIn("score function", query.lower())
        self.assertIn("dot product", query.lower())

    def test_broad_query_hints_adds_relevant_intents(self):
        hints = broad_query_hints("official API benchmark comparison with complexity limitations")

        self.assertIn("api", hints)
        self.assertIn("benchmark", hints)
        self.assertIn("comparison", hints)
        self.assertIn("complexity", hints)

    def test_fallback_gap_queries_do_not_copy_long_gap_text(self):
        queries = fallback_gap_retrieval_queries(
            [
                "Missing Evidence: the report lacks exact implementation details, benchmark numbers, "
                "and full background explanation that should not be copied verbatim into retrieval queries."
            ],
            objective="Research topic",
        )

        self.assertEqual(len(queries), 1)
        self.assertLess(len(queries[0]), 220)
        self.assertNotIn("Missing Evidence", queries[0])
        self.assertIn("benchmark", queries[0].lower())

    def test_complete_sub_question_query_coverage_backfills_skipped_topics(self):
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": [
                "What are major applications and benchmark results of attention mechanisms in NLP and computer vision?",
                "What are known limitations of attention mechanisms and recent efficient attention approaches such as Linformer and Performer?",
            ],
        }

        queries = complete_sub_question_query_coverage(
            ["attention mechanism definition", "attention mechanism equation"],
            plan,
        )
        joined = " ".join(queries).lower()

        self.assertIn("nlp", joined)
        self.assertIn("computer vision", joined)
        self.assertIn("linformer", joined)
        self.assertIn("performer", joined)
        self.assertIn("what source-backed context", joined)
        self.assertNotIn("are major", joined)
        self.assertLessEqual(len(queries), 6)

    def test_complete_sub_question_query_coverage_groups_queries_by_question(self):
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": [
                "What is the Bahdanau additive attention equation?",
                "What benchmark results demonstrate Transformer performance?",
            ],
        }

        queries = complete_sub_question_query_coverage(
            [
                "What benchmark tables report Transformer BLEU scores and performance results?",
                "Which evidence explains Bahdanau additive attention score function equation?",
                "attention mechanism overview",
            ],
            plan,
        )

        self.assertGreaterEqual(len(queries), 4)
        self.assertLessEqual(len(queries), 6)
        self.assertIn("Bahdanau", queries[0])
        self.assertIn("benchmark", " ".join(queries[2:]).lower())

    def test_parse_gap_query_lines_rejects_reasoning_output(self):
        raw = (
            "<think>Analyze user input and plan query generation.</think>\n"
            "G1: scaled dot product attention equation softmax sqrt dk"
        )

        queries = parse_gap_query_lines(raw)

        self.assertEqual(queries, ["scaled dot product attention equation softmax sqrt dk"])

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
        prompt = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("broad, high-recall RAG retrieval queries", prompt)
        self.assertIn("concept/context query", prompt)
        self.assertIn("Do not output only narrow labels", prompt)
        self.assertIn("source/section query", prompt)
        self.assertNotIn('"query 1"', prompt)

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

    def test_select_synthesis_context_prefers_question_planned_sources(self):
        results = [
            RetrievalResult(
                id="secondary",
                document="API API API tutorial overview with enough content for meaningful evidence. " * 4,
                metadata={"title": "Tutorial", "url": "https://example.com/tutorial"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="official",
                document="Official reference covers parameters and usage examples. " * 4,
                metadata={"title": "Official docs", "url": "https://docs.example.com/api"},
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            ),
        ]

        selected = select_synthesis_context(
            results,
            ["What is the API usage?"],
            question_source_urls={"What is the API usage?": ["https://docs.example.com/api"]},
            max_chunks=1,
            per_question=1,
        )

        self.assertEqual([result.id for result in selected], ["official"])

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

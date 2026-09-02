import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.rag.generation import (
    audit_synthesis_citations,
    build_sub_question_evidence_packs,
    build_coverage_by_question,
    build_generation_context,
    browser_signal_results,
    compact_retrieved_chunks,
    coverage_gap_items,
    deterministic_synthesis_from_evidence_packs,
    evidence_focused_question_query,
    fallback_gap_retrieval_queries,
    format_evidence_packs_for_prompt,
    high_signal_browser_snippets,
    parse_gap_query_lines,
    planner_question_source_urls,
    planner_sub_question_specs,
    print_synthesis_chunks,
    report_supporting_chunks,
    retrieve_full_collection_enabled,
    result_supports_question,
    select_synthesis_context,
    synthesize_context_for_report,
    synthesis_quality_issues,
    trim_synthesis_prompt,
)
from src.rag.evidence_spans import supporting_chunks_from_evidence_spans
from src.rag.query_helpers import broad_query_hints, question_required_facets, retrieval_topic_phrase
from src.rag.sub_question_context import (
    browser_question_context_retrieve,
    clean_retrieval_query,
    complete_sub_question_query_coverage,
    is_valid_retrieval_query,
    llm_sub_question_retrieval_query_result,
    parse_llm_retrieval_queries,
    precision_retrieval_queries,
    retrieve_sub_question_context_groups,
    retrieve_sub_question_context,
    select_question_first_synthesis_context,
    sub_question_retrieval_max_workers,
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

    def test_build_sub_question_evidence_packs_prefers_assigned_question_chunk(self):
        results = [
            RetrievalResult(
                id="assigned",
                document="This source-backed passage has enough context for a detailed section. " * 4,
                metadata={
                    "title": "Assigned source",
                    "url": "https://example.com/assigned",
                    "synthesis_question": "What is alpha?",
                },
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="overlap",
                document="Alpha alpha alpha general text with enough evidence content for retrieval. " * 4,
                metadata={"title": "Overlap source", "url": "https://example.com/overlap"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "assigned", "title": "Assigned source", "url": "https://example.com/assigned"},
            {"index": 2, "id": "overlap", "title": "Overlap source", "url": "https://example.com/overlap"},
        ]

        packs = build_sub_question_evidence_packs(["What is alpha?"], results, sources, max_chunks_per_question=1)

        self.assertEqual(packs[0]["chunks"][0]["id"], "assigned")

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

    def test_evidence_pack_prefers_planned_facet_source_over_generic_signal(self):
        question = "What are the key contributions and equations of the AlphaMethod paper?"
        results = [
            RetrievalResult(
                id="generic-method",
                document="General model equation formula variables components benchmark results. " * 4,
                metadata={"title": "Generic method", "url": "https://example.org/generic-method", "source_type": "paper"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="alpha-method",
                document="AlphaMethod paper contribution introduces the alignment model equation a(x,y)=softmax(score(x,y)). " * 3,
                metadata={
                    "title": "AlphaMethod paper",
                    "url": "https://example.org/alpha-method",
                    "source_type": "paper",
                    "synthesis_question": question,
                },
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "generic-method", "url": "https://example.org/generic-method"},
            {"index": 2, "id": "alpha-method", "url": "https://example.org/alpha-method"},
        ]

        packs = build_sub_question_evidence_packs(
            [question],
            results,
            sources,
            question_source_urls={question: ["https://example.org/alpha-method"]},
            max_chunks_per_question=1,
        )

        self.assertEqual(packs[0]["chunks"][0]["id"], "alpha-method")

    def test_evidence_pack_ignores_generic_date_facets_when_exact_evidence_exists(self):
        question = "What are the core equations governing the original method introduced by AlphaMethod et al. (2020)?"
        results = [
            RetrievalResult(
                id="alpha-equation",
                document="AlphaMethod defines the score function as a(x,y)=softmax(score(x,y)), where x and y are model states. " * 3,
                metadata={"title": "AlphaMethod paper", "url": "https://example.org/alpha-method", "source_type": "paper"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "alpha-equation", "url": "https://example.org/alpha-method"}]

        packs = build_sub_question_evidence_packs([question], results, sources)
        coverage = build_coverage_by_question(
            f"{question}\nThe equation is covered [1].",
            [{"question_id": "q001", "question": question, "required_evidence": ["equation"]}],
            sources,
            evidence_packs=packs,
        )

        self.assertNotIn("2020", question_required_facets(question))
        self.assertEqual(packs[0]["coverage"], "covered")
        self.assertEqual(coverage[0]["status"], "covered")
        self.assertFalse(coverage[0]["missing_reason"])

    def test_evidence_pack_uses_chunk_level_signal_for_compound_equation_facets(self):
        question = "What are the equations for Alpha attention and Beta attention?"
        results = [
            RetrievalResult(
                id="compound-equations",
                document=(
                    "Alpha attention and Beta attention are related mechanisms. "
                    "The shared formulation is y = softmax(QK^T / sqrt(d)) V with learned projections. "
                )
                * 3,
                metadata={"title": "Compound equations", "url": "https://example.org/compound"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "compound-equations", "url": "https://example.org/compound"}]

        packs = build_sub_question_evidence_packs([question], results, sources)

        self.assertEqual(packs[0]["coverage"], "covered")
        self.assertEqual(packs[0]["missing_facet_evidence"], [])

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

    def test_evidence_pack_keeps_compound_api_question_partial_until_all_facets_exist(self):
        question = "What official API usage exists in PyTorch and TensorFlow?"
        results = [
            RetrievalResult(
                id="pytorch-api",
                document="PyTorch official API documentation explains the class, parameters, and usage examples. " * 3,
                metadata={"title": "PyTorch docs", "url": "https://pytorch.org/docs/stable/generated/api.html"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "pytorch-api", "url": "https://pytorch.org/docs/stable/generated/api.html"}]

        packs = build_sub_question_evidence_packs([question], results, sources)
        coverage = build_coverage_by_question(
            f"{question}\nThe APIs are covered [1].",
            [{"question_id": "q001", "question": question, "required_evidence": ["api"]}],
            sources,
            evidence_packs=packs,
        )

        self.assertEqual(packs[0]["coverage"], "partial")
        self.assertIn("PyTorch", packs[0]["covered_facets"])
        self.assertIn("TensorFlow", packs[0]["missing_facets"])
        self.assertEqual(coverage[0]["status"], "partial")
        self.assertIn("TensorFlow", coverage[0]["missing_reason"])

    def test_evidence_pack_tracks_missing_listed_method_facets(self):
        question = "What efficient methods such as Linformer, Performer, and Longformer trade off performance and cost?"
        results = [
            RetrievalResult(
                id="linformer",
                document="Linformer reports benchmark accuracy of 91.0% and linear memory complexity. " * 3,
                metadata={"title": "Linformer paper", "url": "https://arxiv.org/pdf/2006.04768"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="longformer",
                document="Longformer benchmark results discuss accuracy, memory cost, and long sequence complexity. " * 3,
                metadata={"title": "Longformer paper", "url": "https://arxiv.org/pdf/2004.05150"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "linformer", "url": "https://arxiv.org/pdf/2006.04768"},
            {"index": 2, "id": "longformer", "url": "https://arxiv.org/pdf/2004.05150"},
        ]

        packs = build_sub_question_evidence_packs([question], results, sources)
        prompt_context = format_evidence_packs_for_prompt(packs)

        self.assertEqual(packs[0]["coverage"], "partial")
        self.assertIn("Performer", packs[0]["missing_facets"])
        self.assertIn("missing facets: Performer", prompt_context)

    def test_evidence_pack_extracts_formula_span(self):
        question = "What equation defines the method?"
        results = [
            RetrievalResult(
                id="formula",
                document="The method equation is y = softmax(x), where x is the input score vector. " * 3,
                metadata={"title": "Formula source", "url": "https://arxiv.org/abs/1234.5678", "has_formula_signal": True},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "formula", "url": "https://arxiv.org/abs/1234.5678"}]

        packs = build_sub_question_evidence_packs([question], results, sources)

        self.assertGreater(packs[0]["evidence_span_count"], 0)
        self.assertEqual(packs[0]["evidence_spans"][0]["evidence_type"], "equation")
        self.assertIn("y = softmax(x)", packs[0]["evidence_spans"][0]["text"])

    def test_evidence_pack_extracts_table_span(self):
        question = "What benchmark table reports BLEU scores?"
        table_json = (
            'Table data JSON:\n{"headers":["Model","BLEU"],'
            '"records":[{"Model":"Transformer","BLEU":"28.4"},{"Model":"Baseline","BLEU":"24.6"}]}'
        )
        results = [
            RetrievalResult(
                id="table",
                document=table_json,
                metadata={"title": "Benchmark table", "url": "https://example.com/results", "chunk_kind": "table", "has_table_signal": True},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        sources = [{"index": 1, "id": "table", "url": "https://example.com/results"}]

        packs = build_sub_question_evidence_packs([question], results, sources)
        span_chunks = supporting_chunks_from_evidence_spans(packs, max_spans_per_question=1)

        self.assertEqual(packs[0]["evidence_spans"][0]["evidence_type"], "table")
        self.assertIn("Table columns: Model, BLEU", packs[0]["evidence_spans"][0]["text"])
        self.assertEqual(span_chunks[0]["chunk_kind"], "evidence_span")
        self.assertIn("Evidence type: table", span_chunks[0]["content"])

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

    def test_build_generation_context_balances_planner_questions(self):
        results = [
            RetrievalResult(
                id=f"alpha-{index}",
                document="Alpha question evidence with enough detail. " * 5,
                metadata={"title": f"Alpha {index}", "url": f"https://example.com/a{index}", "synthesis_question": "What is alpha?"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
            for index in range(4)
        ] + [
            RetrievalResult(
                id="beta-1",
                document="Beta question evidence with enough detail. " * 5,
                metadata={"title": "Beta", "url": "https://example.com/b", "synthesis_question": "What is beta?"},
                score=0.8,
                semantic_score=0.8,
                bm25_score=0.0,
            )
        ]

        context, _sources = build_generation_context(
            results,
            max_context_chars=1800,
            planner_questions=["What is alpha?", "What is beta?"],
        )

        self.assertIn("Alpha", context)
        self.assertIn("Beta", context)

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

    def test_synthesis_quality_flags_too_short_uncited_output(self):
        question = "What benchmark results demonstrate model impact?"
        issues = synthesis_quality_issues(
            "Instruction Coverage Checklist. No retrieved context was available for report synthesis.",
            [question],
            [
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [
                        {
                            "source_index": 1,
                            "content": "The primary paper reports 28.4 BLEU on a benchmark task. " * 3,
                        }
                    ],
                }
            ],
            [{"index": 1}],
        )

        self.assertTrue(any("too short" in issue for issue in issues))
        self.assertTrue(any("no retrieved context" in issue for issue in issues))
        self.assertTrue(any("no source citations" in issue for issue in issues))

    def test_deterministic_synthesis_from_evidence_packs_maps_questions(self):
        question = "What is the official API?"
        synthesis = deterministic_synthesis_from_evidence_packs(
            objective="Library feature",
            synthesis_instruction="Explain API usage.",
            planner_questions=[question],
            evidence_packs=[
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [
                        {
                            "source_index": 2,
                            "title": "Official Docs",
                            "content": "Official API reference lists parameters and usage examples. " * 3,
                        }
                    ],
                }
            ],
        )

        self.assertIn(question, synthesis)
        self.assertIn("[2] Official Docs", synthesis)
        self.assertIn("Recommended Report Structure", synthesis)

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False)
    @patch("src.rag.generation.create_chat_completion_with_retries")
    def test_synthesize_context_for_report_retries_weak_synthesis(self, completion):
        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]
                self.model = "test-model"

        weak = "Instruction Coverage Checklist. No retrieved context was available for report synthesis."
        strong = (
            "1. Instruction Coverage Checklist\n"
            "- What benchmark results demonstrate model impact? Covered [1].\n\n"
            "2. Coverage Map\n"
            "- What benchmark results demonstrate model impact? Covered [1].\n\n"
            "3. Section Notes By Planner Question\n"
            "### What benchmark results demonstrate model impact?\n"
            "- The primary paper reports 28.4 BLEU on a benchmark task [1].\n\n"
            "4. Cross-Source Synthesis\n"
            "- The evidence supports the benchmark claim [1].\n\n"
            "5. Technical Details To Preserve\n"
            "- Preserve 28.4 BLEU [1].\n\n"
            "6. Conflicts Or Gaps\n"
            "- No unresolved gaps.\n\n"
            "7. Recommended Report Structure\n"
            "- Include the benchmark section with [1]."
        )
        completion.side_effect = [Response(weak), Response(strong)]
        context = [
            RetrievalResult(
                id="metric",
                document="The primary paper reports 28.4 BLEU on a benchmark task with enough evidence text. " * 3,
                metadata={"title": "Primary Paper", "url": "https://arxiv.org/pdf/1234.5678", "source_type": "arxiv"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        payload = synthesize_context_for_report(
            objective="Model impact",
            retrieved_context=context,
            planner_questions=["What benchmark results demonstrate model impact?"],
            max_context_chars=9000,
            max_tokens=900,
        )

        self.assertEqual(completion.call_count, 2)
        self.assertEqual(payload["synthesis_attempts"], 2)
        self.assertFalse(payload["synthesis_quality_fallback_used"])
        self.assertIn("28.4 BLEU", payload["synthesis"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False)
    @patch("src.rag.generation.create_chat_completion_with_retries")
    def test_synthesize_context_for_report_falls_back_after_weak_synthesis(self, completion):
        class Message:
            content = "No retrieved context was available for report synthesis."

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            model = "test-model"

        completion.return_value = Response()
        question = "What benchmark results demonstrate model impact?"
        context = [
            RetrievalResult(
                id="metric",
                document="The primary paper reports 28.4 BLEU on a benchmark task with enough evidence text. " * 3,
                metadata={"title": "Primary Paper", "url": "https://arxiv.org/pdf/1234.5678", "source_type": "arxiv"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        payload = synthesize_context_for_report(
            objective="Model impact",
            retrieved_context=context,
            planner_questions=[question],
            max_context_chars=9000,
            max_tokens=900,
        )

        self.assertEqual(completion.call_count, 3)
        self.assertTrue(payload["synthesis_quality_fallback_used"])
        self.assertIn(question, payload["synthesis"])
        self.assertIn("28.4 BLEU", payload["synthesis"])

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

        self.assertIn("benchmark", joined)
        self.assertIn("metrics", joined)
        self.assertFalse(any(query.lower().startswith(("what ", "which ", "where ", "how ")) for query in queries))
        self.assertNotIn("source-backed context", joined)
        self.assertNotIn("overview evidence", joined)
        self.assertNotIn("evidence details examples equations benchmarks limitations", joined)

    def test_sub_question_retrieval_queries_focus_listed_facets(self):
        queries = sub_question_retrieval_queries(
            "What are variants such as additive, multiplicative, and self-attention and how do they differ?",
            objective="Attention mechanism",
        )
        joined = " ".join(queries).lower()

        self.assertEqual(len(queries), 3)
        self.assertTrue(any(query.lower().startswith("additive ") for query in queries))
        self.assertTrue(any(query.lower().startswith("multiplicative ") for query in queries))
        self.assertTrue(any(query.lower().startswith("self-attention ") for query in queries))
        self.assertIn("comparison", joined)
        self.assertNotIn("details examples equations benchmarks limitations", joined)

    def test_valid_retrieval_queries_sanitizes_generic_query_tails(self):
        plan = {"objective": "Attention mechanism", "sub_questions": ["What is attention?"]}

        queries = valid_retrieval_queries(
            [
                "Which evidence gives attention mechanism details examples equations benchmarks limitations",
                "Attention mechanism definition attention mechanism machine learning primary source official docs paper",
            ],
            plan,
        )
        joined = " ".join(queries).lower()

        self.assertTrue(queries)
        self.assertFalse(any(query.lower().startswith(("what ", "which ", "where ", "how ")) for query in queries))
        self.assertNotIn("evidence gives", joined)
        self.assertNotIn("primary source official docs paper", joined)
        self.assertNotIn("details examples equations benchmarks limitations", joined)

    def test_valid_retrieval_queries_repairs_typos_and_removes_unrequested_sources(self):
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": [
                "What is the definition of the attention mechanism in neural networks?",
                "What seminal papers introduced attention mechanisms?",
                "What official APIs exist for TensorFlow and PyTorch?",
            ],
        }

        queries = valid_retrieval_queries(
            [
                "formulation attention menchanism definition mathematical mechanism neural networks purpose",
                "https://www.ibm.com/think/topics/attention-mechanism seminal papers that introduced attention mechanisms",
                "PyTorch Attention menchanism official API layers found Hugging documentation signature",
                "Hugging Face Attention menchanism official API layers found documentation signature",
            ],
            plan,
        )
        joined = " ".join(queries).lower()

        self.assertIn("attention mechanism", joined)
        self.assertNotIn("menchanism", joined)
        self.assertNotIn("http", joined)
        self.assertNotIn("ibm", joined)
        self.assertNotIn("hugging", joined)
        self.assertTrue(any("pytorch" in query.lower() for query in queries))

    def test_sub_question_retrieval_queries_do_not_create_descriptor_only_facets(self):
        queries = sub_question_retrieval_queries(
            "What seminal papers introduced attention mechanisms?",
            objective="Attention mechanism",
        )

        self.assertIn("seminal papers introduced attention mechanisms", queries)
        self.assertNotIn("seminal Attention mechanism", queries)

    def test_clean_retrieval_query_removes_prompt_instruction_terms(self):
        query = clean_retrieval_query(
            "known limitations challenges attention mechanisms Extract source-backed evidence answering authoritative source https"
        )

        self.assertEqual(query, "known limitations challenges attention mechanisms")

    def test_trim_synthesis_prompt_preserves_final_instructions(self):
        prompt = (
            "Research objective:\nX\n\n"
            "Per-question evidence packs:\n"
            + ("evidence pack detail. " * 500)
            + "\n\nRetrieved context from multiple sources:\n"
            + ("retrieved source chunk. " * 800)
            + "\n\nCreate a detailed report-agent-ready evidence package\n"
            "Do not invent source names, authors, dates, titles, papers, benchmark numbers, equations, or citations."
        )

        trimmed = trim_synthesis_prompt(prompt, max_chars=8000)

        self.assertLessEqual(len(trimmed), 8000)
        self.assertIn("Create a detailed report-agent-ready evidence package", trimmed)
        self.assertIn("Do not invent source names", trimmed)
        self.assertIn("Section trimmed to fit synthesis token budget", trimmed)

    def test_format_evidence_packs_for_prompt_can_compact_per_question(self):
        packs = [
            {
                "question": "What is alpha?",
                "coverage": "strong",
                "evidence_spans": [
                    {"source_index": 1, "evidence_type": "definition", "text": "First span"},
                    {"source_index": 2, "evidence_type": "equation", "text": "Second span"},
                ],
                "chunks": [
                    {"source_index": 1, "title": "One", "content": "First chunk"},
                    {"source_index": 2, "title": "Two", "content": "Second chunk"},
                ],
            }
        ]

        text = format_evidence_packs_for_prompt(packs, max_spans_per_question=1, max_chunks_per_question=1)

        self.assertIn("What is alpha?", text)
        self.assertIn("First span", text)
        self.assertIn("First chunk", text)
        self.assertNotIn("Second span", text)
        self.assertNotIn("Second chunk", text)

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
        self.assertNotIn("what source-backed context", joined)
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

    @patch.dict("os.environ", {"RAG_QUERY_REWRITE_PROVIDER": "groq"}, clear=True)
    def test_llm_sub_question_retrieval_query_result_defaults_to_qwen(self):
        result = llm_sub_question_retrieval_query_result(
            {"objective": "X research", "sub_questions": ["What is X?"]},
        )

        self.assertEqual(result["model"], "qwen/qwen3.6-27b")
        self.assertEqual(result["error"], "GROQ_API_KEY is not set")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "RAG_SUBQUESTION_QUERY_MODEL": "llama-test", "RAG_QUERY_REWRITE_PROVIDER": "groq"}, clear=False)
    @patch("src.rag.sub_question_context.create_chat_completion_with_retries")
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

        self.assertEqual(result["queries"], ["x research overview", "x research details"])
        self.assertEqual(result["model"], "llama-test")
        self.assertEqual(result["error"], "")
        self.assertEqual(completion.call_args.kwargs["model"], "llama-test")
        prompt = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("compact, high-recall RAG search queries", prompt)
        self.assertIn("keyword-style queries", prompt)
        self.assertIn("Do not start queries with", prompt)
        self.assertIn("actual planner sub-question topic", prompt)
        self.assertIn('no "extract"', prompt)
        self.assertIn("source-targeted query", prompt)
        self.assertNotIn('"query 1"', prompt)

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "RAG_SUBQUESTION_QUERY_MODEL": "llama-test", "RAG_QUERY_REWRITE_PROVIDER": "groq"}, clear=False)
    @patch("src.rag.sub_question_context.create_chat_completion_with_retries")
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

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "RAG_QUERY_REWRITE_PROVIDER": "auto"}, clear=False)
    @patch("src.rag.sub_question_context.hf_sub_question_retrieval_query_result")
    @patch("src.rag.sub_question_context.create_chat_completion_with_retries")
    def test_llm_sub_question_retrieval_query_result_falls_back_to_hf(self, completion, hf_rewrite):
        completion.side_effect = RuntimeError("Groq failed")
        hf_rewrite.return_value = {
            "queries": ["x research overview", "x research evidence details"],
            "model": "hf-test",
            "error": "",
            "raw_response": "{}",
            "provider": "hf",
            "fallback_reason": "Groq failed",
        }

        result = llm_sub_question_retrieval_query_result(
            {"objective": "X research", "sub_questions": ["What is X?"]},
        )

        self.assertEqual(result["queries"], ["x research overview", "x research evidence details"])
        self.assertEqual(result["provider"], "hf")
        self.assertIn("Groq failed", result["fallback_reason"])

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

    def test_select_synthesis_context_keeps_six_chunks_per_question_by_default(self):
        results = []
        for topic in ("alpha", "beta"):
            for index in range(6):
                results.append(
                    RetrievalResult(
                        id=f"{topic}-{index}",
                        document=f"{topic} topic source-backed details with definitions equations metrics and limitations. " * 3,
                        metadata={"title": f"{topic} source", "url": f"https://example.com/{topic}/{index}"},
                        score=1.0 - (index * 0.01),
                        semantic_score=1.0,
                        bm25_score=0.0,
                    )
                )

        selected = select_synthesis_context(
            results,
            ["What is alpha topic?", "What is beta topic?"],
        )
        selected_ids = {result.id for result in selected}

        self.assertEqual(len([item for item in selected_ids if item.startswith("alpha-")]), 6)
        self.assertEqual(len([item for item in selected_ids if item.startswith("beta-")]), 6)

    def test_select_question_first_synthesis_context_keeps_question_chunks_first(self):
        question_chunks = [
            RetrievalResult(
                id="question-alpha",
                document="Alpha planner question evidence with definitions equations and limitations. " * 4,
                metadata={"title": "Alpha", "url": "https://example.com/alpha", "synthesis_question": "What is alpha?"},
                score=0.4,
                semantic_score=0.4,
                bm25_score=0.0,
            )
        ]
        fallback = [
            RetrievalResult(
                id="browser-signal",
                document="Browser fallback evidence with definitions equations and limitations. " * 4,
                metadata={"title": "Browser", "url": "https://example.com/browser"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        selected = select_question_first_synthesis_context(
            question_context_results=question_chunks,
            fallback_context=fallback,
            planner_questions=["What is alpha?"],
        )

        self.assertEqual(selected[0].id, "question-alpha")
        self.assertIn("browser-signal", {result.id for result in selected})

    def test_retrieve_sub_question_context_reranks_candidates_and_keeps_six(self):
        calls = []

        def fake_retrieve(**kwargs):
            calls.append(kwargs)
            topic = "beta" if "beta" in " ".join(kwargs["queries"]).lower() else "alpha"
            return [
                RetrievalResult(
                    id=f"{topic}-{index}",
                    document=f"{topic} topic source-backed details with definitions equations metrics and limitations. " * 3,
                    metadata={"title": f"{topic} source", "url": f"https://example.com/{topic}/{index}"},
                    score=1.0 - (index * 0.01),
                    semantic_score=1.0,
                    bm25_score=0.0,
                )
                for index in range(20)
            ]

        with patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", side_effect=fake_retrieve):
            selected = retrieve_sub_question_context(
                research_plan={},
                questions=["What is alpha topic?", "What is beta topic?"],
                objective="Alpha objective",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=20,
                final_chunks=6,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=True,
                reranker_model="cross-encoder",
                rerank_k=20,
                rerank_weight=0.7,
            )

        selected_ids = {result.id for result in selected}

        self.assertEqual(len(calls), 2)
        self.assertEqual(len([item for item in selected_ids if item.startswith("alpha-")]), 6)
        self.assertEqual(len([item for item in selected_ids if item.startswith("beta-")]), 6)
        self.assertTrue(all(call["top_k"] == 20 for call in calls))
        self.assertTrue(all(call["per_query_k"] == 20 for call in calls))
        self.assertTrue(all(call["rerank"] for call in calls))
        self.assertTrue(all(result.metadata.get("synthesis_question") for result in selected))

    def test_retrieve_sub_question_context_groups_exposes_counts(self):
        def fake_retrieve(**kwargs):
            return [
                RetrievalResult(
                    id=f"alpha-{index}",
                    document="Alpha topic source-backed details with definitions equations metrics and limitations. " * 3,
                    metadata={"title": "Alpha source", "url": f"https://example.com/alpha/{index}"},
                    score=1.0 - (index * 0.01),
                    semantic_score=1.0,
                    bm25_score=0.0,
                )
                for index in range(8)
            ]

        with patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", side_effect=fake_retrieve):
            groups = retrieve_sub_question_context_groups(
                research_plan={},
                questions=["What is alpha topic?"],
                objective="Alpha objective",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=8,
                final_chunks=3,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=False,
                reranker_model="cross-encoder",
                rerank_k=8,
                rerank_weight=0.7,
            )

        self.assertEqual(groups[0]["candidate_count"], 8)
        self.assertEqual(groups[0]["chunk_count"], 3)
        self.assertEqual(groups[0]["chunks"][0].metadata["synthesis_question"], "What is alpha topic?")

    def test_retrieve_sub_question_context_groups_prefers_planner_source_url(self):
        question = "What equation defines AlphaMethod?"
        planned = [
            RetrievalResult(
                id="planned-alpha",
                document="AlphaMethod official paper defines the equation y = softmax(x) with variables. " * 3,
                metadata={"title": "AlphaMethod paper", "url": "https://example.org/alpha"},
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            )
        ]
        hybrid = [
            RetrievalResult(
                id="generic-alpha",
                document="AlphaMethod tutorial mentions equation formula variables in a general overview. " * 3,
                metadata={"title": "Generic AlphaMethod", "url": "https://example.org/generic"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]
        plan = {
            "objective": "AlphaMethod",
            "sub_questions": [question],
            "tasks": [{"query_context": "q001", "url": "https://example.org/alpha"}],
        }

        with (
            patch("src.rag.sub_question_context.source_url_coverage_retrieve", return_value=planned) as source_retrieve,
            patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", return_value=hybrid),
        ):
            groups = retrieve_sub_question_context_groups(
                research_plan=plan,
                questions=[question],
                objective="AlphaMethod",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=8,
                final_chunks=1,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=False,
                reranker_model="cross-encoder",
                rerank_k=8,
                rerank_weight=0.7,
            )

        source_retrieve.assert_called_once()
        self.assertIn("source_url", groups[0]["fallback_sources"])
        self.assertEqual(groups[0]["chunks"][0].id, "planned-alpha")
        self.assertEqual(groups[0]["chunks"][0].metadata["synthesis_question"], question)

    def test_retrieve_sub_question_context_groups_rescues_missing_facets(self):
        question = "What efficient methods such as Linformer, Performer, and Longformer reduce attention cost?"
        initial = [
            RetrievalResult(
                id="linformer",
                document="Linformer reduces attention complexity with low-rank projections. " * 4,
                metadata={"title": "Linformer", "url": "https://arxiv.org/pdf/2006.04768"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="longformer",
                document="Longformer uses sparse local and global attention for long sequence complexity. " * 4,
                metadata={"title": "Longformer", "url": "https://arxiv.org/pdf/2004.05150"},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="generic",
                document="Efficient attention methods reduce memory and runtime for long contexts. " * 4,
                metadata={"title": "Generic", "url": "https://example.com/generic"},
                score=0.8,
                semantic_score=0.8,
                bm25_score=0.0,
            ),
        ]
        rescued = [
            RetrievalResult(
                id="performer",
                document="Performer approximates softmax attention with random features for linear attention complexity. " * 4,
                metadata={"title": "Performer", "url": "https://arxiv.org/pdf/2009.14794"},
                score=0.7,
                semantic_score=0.7,
                bm25_score=0.0,
            )
        ]

        with (
            patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", return_value=initial),
            patch("src.rag.sub_question_context.facet_rescue_context_retrieve", return_value=rescued) as facet_scan,
        ):
            groups = retrieve_sub_question_context_groups(
                research_plan={"sub_questions": [question]},
                questions=[question],
                objective="Attention mechanism",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=8,
                final_chunks=3,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=False,
                reranker_model="cross-encoder",
                rerank_k=8,
                rerank_weight=0.7,
            )

        selected_ids = {result.id for result in groups[0]["chunks"]}

        facet_scan.assert_called_once()
        self.assertIn("facet_scan", groups[0]["fallback_sources"])
        self.assertEqual({"linformer", "longformer", "performer"}, selected_ids)

    def test_retrieve_sub_question_context_groups_scans_collection_when_hybrid_empty(self):
        class FakeCollection:
            def get(self, **kwargs):
                return {
                    "ids": ["weak", "formula", "primary"],
                    "documents": [
                        "Unrelated background with enough words to be meaningful evidence. " * 4,
                        "Alpha topic equation formula variables and detailed source-backed explanation. " * 4,
                        "Official alpha source explains definition equation formula and implementation. " * 4,
                    ],
                    "metadatas": [
                        {"title": "Weak", "url": "https://example.com/weak"},
                        {"title": "Formula", "url": "https://example.com/formula", "has_formula_signal": True},
                        {"title": "Primary", "url": "https://arxiv.org/pdf/1234.5678", "source_type": "arxiv"},
                    ],
                }

        with (
            patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", return_value=[]),
            patch("src.rag.sub_question_context.get_collection", return_value=FakeCollection()),
            patch("src.rag.sub_question_context.expand_parent_context_results", side_effect=lambda results, chroma_path: list(results)),
        ):
            groups = retrieve_sub_question_context_groups(
                research_plan={},
                questions=["What is alpha topic equation?"],
                objective="Alpha objective",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=8,
                final_chunks=2,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=False,
                reranker_model="cross-encoder",
                rerank_k=8,
                rerank_weight=0.7,
            )

        self.assertGreaterEqual(groups[0]["candidate_count"], 2)
        self.assertEqual(groups[0]["chunk_count"], 2)
        self.assertTrue(all(chunk.metadata.get("synthesis_question") for chunk in groups[0]["chunks"]))
        self.assertEqual(groups[0]["chunks"][0].id, "primary")

    def test_retrieve_sub_question_context_groups_uses_browser_results_when_index_empty(self):
        browser_results = [
            {
                "task_id": "task_001",
                "query_context": "q001",
                "sources": [
                    {
                        "url": "https://arxiv.org/pdf/1508.04025",
                        "title": "Effective Approaches to Attention-based Neural Machine Translation",
                        "source_type": "pdf",
                        "source_quality": "useful_primary",
                        "source_authority": "primary",
                        "full_content": (
                            "Luong multiplicative attention defines global attention and local attention. "
                            "The general score function uses a bilinear equation between decoder state "
                            "and encoder hidden state. This source compares multiplicative attention "
                            "with additive attention and reports WMT translation evidence. "
                        ) * 4,
                    }
                ],
            }
        ]
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": ["How does multiplicative Luong attention work?"],
            "sub_question_specs": [{"question_id": "q001", "question": "How does multiplicative Luong attention work?"}],
            "tasks": [{"query_context": "q001", "url": "https://arxiv.org/abs/1508.04025"}],
        }

        with (
            patch("src.rag.sub_question_context.multi_query_hybrid_retrieve", return_value=[]),
            patch("src.rag.sub_question_context.source_url_coverage_retrieve", return_value=[]),
            patch("src.rag.sub_question_context.collection_scan_question_retrieve", return_value=[]),
        ):
            groups = retrieve_sub_question_context_groups(
                research_plan=plan,
                questions=plan["sub_questions"],
                objective="Attention mechanism",
                chroma_path="/tmp/chroma",
                collection_name="test",
                history_keys=[],
                candidate_chunks=8,
                final_chunks=2,
                per_query_k=25,
                semantic_k=10,
                bm25_k=10,
                semantic_weight=0.3,
                bm25_weight=0.3,
                authority_weight=0.4,
                bm25_scan_limit=100,
                embedding_device="",
                rerank=False,
                reranker_model="cross-encoder",
                rerank_k=8,
                rerank_weight=0.7,
                browser_results=browser_results,
            )

        self.assertGreaterEqual(groups[0]["candidate_count"], 1)
        self.assertGreaterEqual(groups[0]["chunk_count"], 1)
        self.assertIn("browser_results", groups[0]["fallback_sources"])
        self.assertEqual(groups[0]["chunks"][0].metadata["synthesis_question"], "How does multiplicative Luong attention work?")
        self.assertEqual(groups[0]["chunks"][0].metadata["url"], "https://arxiv.org/pdf/1508.04025")

    def test_browser_question_context_prefers_exact_benchmark_from_primary_source(self):
        browser_results = [
            {
                "sources": [
                    {
                        "url": "https://example.com/benchmark-overview",
                        "title": "Benchmark Overview",
                        "source_type": "webpage",
                        "full_content": (
                            "This overview discusses benchmark evaluation and model performance in broad terms. "
                            "It does not provide exact metric values for the target source. "
                        ) * 8,
                    },
                    {
                        "url": "https://arxiv.org/pdf/1706.03762",
                        "title": "Attention Is All You Need",
                        "source_type": "pdf",
                        "full_content": (
                            "The Transformer uses attention mechanisms for sequence transduction. "
                            "Experiments on two machine translation tasks show superior quality. "
                            "The model achieves 28.4 BLEU on the WMT 2014 English-to-German task, "
                            "improving over the existing best results by over 2 BLEU. "
                        ) * 4,
                    },
                ]
            }
        ]

        results = browser_question_context_retrieve(
            question="What are the primary applications and benchmark results in NLP and computer vision?",
            queries=["applications benchmark results NLP computer vision"],
            browser_results=browser_results,
            top_k=2,
        )

        self.assertEqual(results[0].metadata["url"], "https://arxiv.org/pdf/1706.03762")
        self.assertIn("28.4 BLEU", results[0].document)
        self.assertIn("WMT 2014", results[0].document)

    def test_browser_question_context_promotes_exact_metric_over_generic_overlap(self):
        browser_results = [
            {
                "sources": [
                    {
                        "url": "https://aclanthology.org/paper.pdf",
                        "title": "Survey",
                        "source_type": "pdf",
                        "source_quality": "useful_authoritative",
                        "full_content": (
                            "Attention mechanisms have common applications in NLP and computer vision. "
                            "The paper discusses benchmark results, performance, and limitations broadly. "
                        ) * 12,
                    },
                    {
                        "url": "https://arxiv.org/pdf/1706.03762",
                        "title": "Primary Paper",
                        "source_type": "pdf",
                        "source_quality": "useful_primary",
                        "full_content": (
                            "The model uses attention mechanisms for machine translation. "
                            "It achieves 28.4 BLEU on the WMT 2014 English-to-German task "
                            "and 41.8 BLEU on English-to-French. "
                        ) * 4,
                    },
                ]
            }
        ]

        results = browser_question_context_retrieve(
            question="What are common applications and benchmark results in NLP and computer vision?",
            queries=["applications benchmark results NLP computer vision"],
            browser_results=browser_results,
            top_k=1,
        )

        self.assertEqual(results[0].metadata["url"], "https://arxiv.org/pdf/1706.03762")
        self.assertIn("28.4 BLEU", results[0].document)

    def test_evidence_packs_promote_exact_primary_benchmark(self):
        results = [
            RetrievalResult(
                id="assigned-overview",
                document=(
                    "Attention mechanisms have common applications in NLP and computer vision. "
                    "This assigned overview mentions benchmark results and performance broadly. "
                )
                * 4,
                metadata={
                    "title": "Assigned overview",
                    "url": "https://example.com/overview",
                    "synthesis_question": "What are common applications and benchmark results in NLP and computer vision?",
                },
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="primary-metric",
                document=(
                    "The Transformer reports machine translation benchmark results: "
                    "28.4 BLEU on the WMT 2014 English-to-German task and 41.8 BLEU on English-to-French. "
                )
                * 3,
                metadata={"title": "Primary paper", "url": "https://arxiv.org/pdf/1706.03762", "source_type": "arxiv"},
                score=0.2,
                semantic_score=0.2,
                bm25_score=0.0,
            ),
        ]
        sources = [
            {"index": 1, "id": "assigned-overview", "url": "https://example.com/overview"},
            {"index": 2, "id": "primary-metric", "url": "https://arxiv.org/pdf/1706.03762"},
        ]

        packs = build_sub_question_evidence_packs(
            ["What are common applications and benchmark results in NLP and computer vision?"],
            results,
            sources,
            max_chunks_per_question=1,
        )

        self.assertEqual(packs[0]["chunks"][0]["id"], "primary-metric")

    def test_sub_question_retrieval_max_workers_is_bounded(self):
        self.assertEqual(sub_question_retrieval_max_workers(10, rerank=False), 4)
        self.assertEqual(sub_question_retrieval_max_workers(10, rerank=True), 2)

        with patch.dict("os.environ", {"RAG_SUBQUESTION_RETRIEVAL_MAX_WORKERS": "3"}):
            self.assertEqual(sub_question_retrieval_max_workers(10, rerank=True), 3)
            self.assertEqual(sub_question_retrieval_max_workers(2, rerank=True), 2)

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

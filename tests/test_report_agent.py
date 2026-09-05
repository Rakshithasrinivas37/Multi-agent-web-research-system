import unittest

import src.agents.report_agent as report_agent_module
from src.agents.report_agent import (
    DEFAULT_REPORT_PROMPT_CHARS,
    DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
    apply_report_evidence_pack_repairs,
    clean_markdown,
    build_report_prompt,
    compact_markdown_at_sentence,
    dedupe_sources,
    evidence_pack_questions,
    format_evidence_packs,
    format_question_coverage,
    format_question_focused_evidence,
    format_single_question_synthesis,
    format_report_revision_feedback,
    format_report_section_outline,
    format_supporting_evidence,
    generate_single_report,
    missing_evidence_constraints,
    missing_sub_question_coverage,
    normalize_final_report,
    normalize_markdown_headings,
    remove_unavailable_citation_markers,
    report_context_gap_items,
    report_context_gap_queries,
    report_evidence_gap_contradictions,
    report_per_question_synthesis_citation_gaps,
    report_generation_token_cap,
    report_quality_issues,
    report_synthesis_gap_contradictions,
    report_needs_revision,
    report_pack_citation_gaps,
    repair_report_by_sections,
    report_schema_issues,
    report_self_critique,
    report_sub_question_coverage_check,
    resolve_report_coverage,
    rewrite_missing_sub_question_queries,
    sources_with_browser_results,
    slugify_filename,
    synthesis_coverage_gap_questions,
    trim_report_prompt,
    unsupported_benchmark_metrics,
)


class ReportAgentTests(unittest.TestCase):
    def test_clean_markdown_strips_thinking_blocks(self):
        text = "Useful.\n<think>hidden reasoning</think>\nFinal."

        self.assertEqual(clean_markdown(text), "Useful.\n\nFinal.")

    def test_normalize_final_report_removes_bad_citations_and_adds_references(self):
        report = """# Topic

## Executive Summary
Supported [1]. Unsupported [9].
"""
        sources = [{"index": 1, "url": "https://example.com/one"}]

        normalized = normalize_final_report(report, sources)

        self.assertIn("Supported [1]. Unsupported", normalized)
        self.assertNotIn("[9]", normalized)
        self.assertIn("## References", normalized)
        self.assertIn("[1] https://example.com/one", normalized)

    def test_report_quality_flags_unavailable_citations(self):
        issues = report_quality_issues(
            "Claim [2].\n\n## References\n[2] https://bad",
            [{"index": 1, "url": "https://example.com"}],
        )

        self.assertIn("report uses unavailable citations: [2]", issues)

    def test_report_quality_flags_placeholder_source_markers(self):
        issues = report_quality_issues(
            "Claim [— benchmark evidence missing].\n\n## References\n[1] https://example.com",
            [{"index": 1, "url": "https://example.com"}],
        )

        self.assertIn("report contains placeholder or non-source citation markers", issues)

    def test_normalize_markdown_headings_removes_duplicate_heading_markers(self):
        markdown = "### ## 1. Definition\nText."

        self.assertEqual(normalize_markdown_headings(markdown), "### 1. Definition\nText.")

    def test_report_quality_flags_benchmark_metrics_missing_from_evidence(self):
        issues = report_quality_issues(
            "## Benchmarks\nViT reaches 77% top-1 accuracy [1].\n\n## References\n[1] https://example.com",
            [{"index": 1, "url": "https://example.com"}],
            evidence_text="Vision Transformer image classification evidence without exact scores.",
        )

        self.assertIn("report includes benchmark metrics not present in evidence: 77%", issues)

    def test_unsupported_benchmark_metrics_allows_evidence_numbers(self):
        report = "BLEU-4 improves from 0.386 to 0.482 [1]."
        evidence = "The evidence says BLEU-4 improves from 0.386 to 0.482."

        self.assertEqual(unsupported_benchmark_metrics(report, evidence), [])

    def test_unsupported_benchmark_metrics_ignores_publication_years(self):
        report = "The 2017 Transformer paper reports WMT benchmark results with BLEU gains [1]."
        evidence = "WMT benchmark evidence includes BLEU gains."

        self.assertEqual(unsupported_benchmark_metrics(report, evidence), [])

    def test_unsupported_benchmark_metrics_ignores_model_name_numbers(self):
        report = "The benchmark compares a model-family-152 baseline with another architecture [1]."
        evidence = "Benchmark evidence compares baseline architectures without exact scores."

        self.assertEqual(unsupported_benchmark_metrics(report, evidence), [])

    def test_report_quality_accepts_references_after_markdown_cleanup(self):
        issues = report_quality_issues(
            "### Executive Summary\nClaim [1].\n\n## References\n[1] https://example.com",
            [{"index": 1, "url": "https://example.com"}],
        )

        self.assertNotIn("report must include a References section", issues)

    def test_dedupe_sources_preserves_existing_source_indexes(self):
        sources = dedupe_sources(
            [
                {"index": 3, "url": "https://paper.example"},
                {"url": "https://new.example"},
                {"index": 14, "url": "https://blog.example"},
            ]
        )

        self.assertEqual([source["index"] for source in sources], [3, 1, 14])

    def test_sources_with_browser_results_preserves_synthesis_indexes(self):
        sources = [
            {"index": 1, "url": "https://paper.example"},
            {"index": 7, "url": "https://paper.example"},
        ]
        browser_results = [
            {
                "sources": [
                    {"title": "Paper duplicate", "url": "https://paper.example"},
                    {"title": "Docs", "url": "https://docs.example.com/api"},
                ]
            }
        ]

        merged = sources_with_browser_results(sources, browser_results)

        self.assertEqual([source["index"] for source in merged], [1, 7, 8])
        self.assertEqual(merged[1]["url"], "https://paper.example")
        self.assertEqual(merged[2]["url"], "https://docs.example.com/api")

    def test_format_report_section_outline_uses_topic_headings(self):
        outline = format_report_section_outline(
            [
                "What benchmark results demonstrate GLUE and WMT performance?",
                "How is PyTorch MultiheadAttention used?",
            ]
        )

        self.assertIn("## 1. Benchmark Demonstrate GLUE And WMT Performance", outline)
        self.assertIn("## 2. PyTorch MultiheadAttention Used", outline)

    def test_build_report_prompt_requires_core_equation_for_formula_questions(self):
        prompt = build_report_prompt(
            objective="Attention mechanism",
            output_format="report",
            planner_questions=["What is the main attention equation and its components?"],
            synthesis="The evidence includes Attention(Q,K,V)=softmax(score(Q,K))V.",
            evidence="[1] Attention equation evidence.",
            sources=[{"index": 1, "url": "https://example.com"}],
        )

        self.assertIn("Core equation", prompt)
        self.assertIn("show the most general/source-backed equation first", prompt)

    def test_build_report_prompt_includes_structured_coverage_contract(self):
        prompt = build_report_prompt(
            objective="Attention mechanism",
            output_format="report",
            planner_questions=["What benchmark results show performance?"],
            synthesis="Benchmark notes are incomplete.",
            evidence="[1] General benchmark source.",
            sources=[{"index": 1, "url": "https://example.com"}],
            coverage_by_question=[
                {
                    "question_id": "q001",
                    "question": "What benchmark results show performance?",
                    "required_evidence": ["benchmark"],
                    "status": "missing",
                    "source_indexes": [],
                }
            ],
        )

        self.assertIn("Synthesis coverage by planner question", prompt)
        self.assertIn("q001: missing", prompt)
        self.assertIn("write a short evidence-gap subsection instead of inventing an answer", prompt)
        self.assertIn("do not include formulas, API names, benchmark values, examples, or detailed explanations", prompt)

    def test_build_report_prompt_includes_evidence_packs(self):
        prompt = build_report_prompt(
            objective="Research topic",
            output_format="report",
            planner_questions=["How is the API used?"],
            synthesis="API usage is covered.",
            evidence="[1] API evidence.",
            sources=[{"index": 1, "url": "https://docs.example.com/api"}],
            evidence_packs=[
                {
                    "question": "How is the API used?",
                    "coverage": "covered",
                    "chunks": [{"source_index": 1, "title": "API docs", "content": "The API usage is documented."}],
                }
            ],
        )

        self.assertIn("Per-question evidence packs", prompt)
        self.assertIn("covered: How is the API used?", prompt)
        self.assertIn("Covered packs must be explained", prompt)
        self.assertIn("Never label a covered evidence pack as an evidence gap", prompt)

    def test_build_report_prompt_keeps_full_evidence_and_synthesis(self):
        evidence_tail = "retrieved evidence tail should remain"
        synthesis_tail = "synthesis tail should remain"
        prompt = build_report_prompt(
            objective="Large research topic",
            output_format="report",
            planner_questions=["What evidence should be covered?"],
            synthesis=("Synthesis sentence. " * 1000) + synthesis_tail,
            evidence=("[1] Evidence sentence. " * 1000) + evidence_tail,
            sources=[{"index": 1, "url": "https://example.com"}],
            evidence_packs=[
                {
                    "question": "What evidence should be covered?",
                    "coverage": "covered",
                    "chunks": [{"source_index": 1, "title": "Source", "content": "Useful evidence. " * 200}],
                }
            ],
        )

        self.assertIn(evidence_tail, prompt)
        self.assertIn(synthesis_tail, prompt)
        self.assertIn("Grounding requirement (strict - read this first)", prompt)
        self.assertLess(prompt.index("Grounding requirement"), prompt.index("Per-question evidence packs"))

    def test_build_report_prompt_can_compact_for_retry(self):
        prompt = build_report_prompt(
            objective="Large research topic",
            output_format="report",
            planner_questions=["What evidence should be covered?"],
            synthesis="Synthesis sentence. " * 1000,
            evidence="[1] Evidence sentence. " * 1000,
            sources=[{"index": 1, "url": "https://example.com"}],
            evidence_packs=[
                {
                    "question": "What evidence should be covered?",
                    "coverage": "covered",
                    "chunks": [{"source_index": 1, "title": "Source", "content": "Useful evidence. " * 200}],
                }
            ],
            compact=True,
        )

        self.assertLessEqual(len(prompt), DEFAULT_REPORT_PROMPT_CHARS)
        self.assertIn("Grounding requirement (strict - read this first)", prompt)

    def test_trim_report_prompt_limits_prompt_size(self):
        prompt = "Rules first.\n\n" + ("long evidence " * 2000)

        trimmed = trim_report_prompt(prompt, max_chars=1000)

        self.assertLessEqual(len(trimmed), 1000)
        self.assertTrue(trimmed.startswith("Rules first."))

    def test_format_question_coverage_lists_status_and_sources(self):
        coverage = format_question_coverage([
            {
                "question_id": "q002",
                "question": "How is the API used?",
                "required_evidence": ["api"],
                "status": "covered",
                "source_indexes": [1, 3],
            }
        ])

        self.assertIn("q002: covered", coverage)
        self.assertIn("sources=[1], [3]", coverage)

    def test_format_evidence_packs_lists_chunks_and_questions(self):
        packs = [
            {
                "question": "What benchmark result is reported?",
                "coverage": "partial",
                "chunks": [{"source_index": 2, "title": "Benchmark", "content": "Metric evidence appears here."}],
            }
        ]

        formatted = format_evidence_packs(packs)

        self.assertIn("partial: What benchmark result is reported?", formatted)
        self.assertIn("[2] Benchmark", formatted)
        self.assertEqual(evidence_pack_questions(packs), ["What benchmark result is reported?"])

    def test_format_evidence_packs_marks_formula_evidence_as_usable(self):
        formatted = format_evidence_packs(
            [
                {
                    "question": "What are the equations of additive attention?",
                    "coverage": "partial",
                    "chunks": [
                        {
                            "source_index": 2,
                            "title": "Bahdanau paper",
                            "content": "The alignment model is a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj).",
                        }
                    ],
                }
            ]
        )

        self.assertIn("Use cited evidence from [2]", formatted)
        self.assertIn("Formula/equation evidence is present", formatted)

    def test_format_supporting_evidence_uses_chunks_once(self):
        context = {
            "supporting_chunks": [
                {
                    "source_index": 1,
                    "title": "Paper",
                    "url": "https://example.com",
                    "content": "Attention evidence from a source.",
                },
                {
                    "source_index": 1,
                    "title": "Paper",
                    "url": "https://example.com",
                    "content": "Attention evidence from a source.",
                },
            ]
        }

        evidence = format_supporting_evidence(context)

        self.assertEqual(evidence.count("Attention evidence"), 1)
        self.assertIn("[1]", evidence)

    def test_format_supporting_evidence_keeps_full_chunk_content(self):
        tail = "important formula appears at the end"
        context = {
            "retrieved_chunks": [
                {
                    "source_index": 2,
                    "title": "Long source",
                    "url": "https://example.com/long",
                    "content": f"{'context evidence ' * 700}{tail}",
                }
            ]
        }

        evidence = format_supporting_evidence(context)

        self.assertIn(tail, evidence)

    def test_format_evidence_packs_keeps_all_chunk_content(self):
        first_tail = "first chunk tail"
        second_tail = "second chunk tail"
        formatted = format_evidence_packs(
            [
                {
                    "question": "What evidence is available?",
                    "coverage": "covered",
                    "chunks": [
                        {"source_index": 1, "title": "One", "content": f"{'alpha ' * 100}{first_tail}"},
                        {"source_index": 2, "title": "Two", "content": f"{'beta ' * 100}{second_tail}"},
                    ],
                }
            ]
        )

        self.assertIn(first_tail, formatted)
        self.assertIn(second_tail, formatted)

    def test_format_question_focused_evidence_keeps_per_question_details(self):
        context = {
            "planner_questions": [
                "What is the equation?",
                "What benchmark result is reported?",
            ],
            "retrieved_chunks": [
                {
                    "source_index": 1,
                    "title": "Paper",
                    "url": "https://paper.example",
                    "content": "General background without the requested detail.",
                },
                {
                    "source_index": 2,
                    "title": "Equation paper",
                    "url": "https://equation.example",
                    "content": "The core formula is score(q,k)=exp(q k) and alpha=sum weights.",
                },
                {
                    "source_index": 3,
                    "title": "Benchmark paper",
                    "url": "https://benchmark.example",
                    "content": "The benchmark result improves BLEU from 20.1 to 24.3.",
                },
            ],
        }

        evidence = format_question_focused_evidence(context, context["planner_questions"])

        self.assertIn("Question: What is the equation?", evidence)
        self.assertIn("score(q,k)=exp(q k)", evidence)
        self.assertIn("Question: What benchmark result is reported?", evidence)
        self.assertIn("BLEU from 20.1 to 24.3", evidence)

    def test_generate_single_report_sends_full_prompt(self):
        captured = {}

        class Message:
            content = "## Executive Summary\nDone.\n\n## References\n"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            model = "test-model"

        def fake_completion(client, **kwargs):
            captured["prompt"] = kwargs["messages"][1]["content"]
            return Response()

        original = report_agent_module.create_chat_completion_with_retries
        report_agent_module.create_chat_completion_with_retries = fake_completion
        try:
            tail = "prompt tail should remain"
            generate_single_report(object(), "test-model", f"{'prompt content ' * 1200}{tail}")
        finally:
            report_agent_module.create_chat_completion_with_retries = original

        self.assertIn(tail, captured["prompt"])

    def test_generate_single_report_retries_with_compact_prompt_on_context_error(self):
        calls = []

        class Message:
            content = "## Executive Summary\nDone.\n\n## References\n"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]
            model = "test-model"

        def fake_completion(client, **kwargs):
            calls.append(kwargs["messages"][1]["content"])
            if len(calls) == 1:
                raise RuntimeError("context_length_exceeded: Please reduce the length of the messages or completion.")
            return Response()

        original = report_agent_module.create_chat_completion_with_retries
        report_agent_module.create_chat_completion_with_retries = fake_completion
        try:
            generate_single_report(object(), "test-model", "full prompt", fallback_prompt="compact prompt")
        finally:
            report_agent_module.create_chat_completion_with_retries = original

        self.assertEqual(calls, ["full prompt", "compact prompt"])

    def test_missing_sub_question_coverage_flags_missing_topic(self):
        report = "The report defines attention and explains scoring."
        questions = [
            "What is the definition of attention?",
            "How is PyTorch MultiheadAttention used?",
        ]

        self.assertEqual(missing_sub_question_coverage(report, questions), [questions[1]])

    def test_report_sub_question_coverage_check_is_inspectable(self):
        questions = [
            "What benchmark results demonstrate GLUE and WMT performance?",
            "How is PyTorch MultiheadAttention used?",
        ]
        report = "This section covers GLUE benchmark and WMT performance evidence in detail."

        check = report_sub_question_coverage_check(report, questions)

        self.assertEqual(check["total"], 2)
        self.assertEqual(check["covered_count"], 1)
        self.assertEqual(check["missing"], [questions[1]])

    def test_report_coverage_does_not_fail_for_synthesis_gap_statuses(self):
        questions = ["What benchmark results demonstrate GLUE and WMT performance?"]
        report = "This report mentions GLUE, WMT, and benchmark performance."

        check = report_sub_question_coverage_check(report, questions)

        self.assertEqual(check["covered_count"], 1)
        self.assertEqual(check["missing"], [])

    def test_missing_sub_question_coverage_accepts_technical_equation_answer(self):
        report = """
## Scaled Dot-Product Attention
The scaled dot-product attention equation is
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.
Here Q is the query matrix, K is the key matrix, V is the value matrix,
and d_k is the key dimension used for scaling.
"""
        questions = ["What is the scaled dot‑product attention equation and its components?"]

        self.assertEqual(missing_sub_question_coverage(report, questions), [])

    def test_report_schema_issues_detects_missing_sections(self):
        report = """# Topic

## Executive Summary
Summary.

## Introduction and Context
Context.

## References
No cited source markers were used.
"""

        issues = report_schema_issues(report, ["What benchmark results demonstrate GLUE and WMT performance?"])

        self.assertIn("missing schema section: cross-cutting analysis/synthesis", issues)
        self.assertIn("missing planner topic section: Benchmark Demonstrate GLUE And WMT Performance", issues)

    def test_report_schema_accepts_nested_markdown_headings(self):
        report = """# Topic

### 1. Executive Summary
Summary.

### 2. Introduction and Context
Context.

### 3. Benchmark Demonstrate GLUE And WMT Performance
Benchmarks.

```python
# This is not a heading
```

### 4. Cross-cutting Analysis and Synthesis
Synthesis.

### 5. Limitations and Open Questions
Limits.

### 6. Conclusion
Done.

## References
[1] https://example.com
"""

        issues = report_schema_issues(report, ["What benchmark results demonstrate GLUE and WMT performance?"])

        self.assertEqual(issues, [])

    def test_report_schema_accepts_concise_topic_heading_for_long_question(self):
        report = """# Topic

## Executive Summary
Summary.

## Introduction and Context
Context.

### Benchmark Evidence of Performance Impact
Benchmarks.

## Cross-cutting Analysis and Synthesis
Synthesis.

## Limitations and Open Questions
Limits.

## Conclusion
Done.

## References
[1] https://example.com
"""

        issues = report_schema_issues(
            report,
            ["What benchmark results demonstrate the performance impact of attention mechanisms on tasks such as machine translation and GLUE?"],
        )

        self.assertEqual(issues, [])

    def test_clean_markdown_strips_open_thinking_block(self):
        text = "Useful.\n<think>hidden reasoning that never closes"

        self.assertEqual(clean_markdown(text), "Useful.")

    def test_report_self_critique_combines_diagnostics(self):
        critique = report_self_critique(
            ["report issue"],
            {"missing": ["What benchmark results demonstrate GLUE and WMT performance?"]},
            ["schema issue"],
        )

        self.assertIn("report issue", critique["unresolved_issues"])
        self.assertIn("schema issue", critique["unresolved_issues"])
        self.assertIn("missing planner topic: What benchmark results demonstrate GLUE and WMT performance?", critique["unresolved_issues"])

    def test_missing_evidence_constraints_extracts_gap_lines(self):
        synthesis = "Covered item.\n- Missing: exact WMT BLEU result is not present."

        self.assertEqual(missing_evidence_constraints(synthesis), ["- Missing: exact WMT BLEU result is not present."])

    def test_report_context_gap_items_and_queries(self):
        context = {"synthesis": "Benchmark evidence is Missing. No GLUE numbers are present."}
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": ["What benchmark results demonstrate GLUE performance?"],
        }

        gaps = report_context_gap_items(context, plan)
        queries = report_context_gap_queries(context, plan)

        self.assertEqual(gaps, plan["sub_questions"])
        self.assertIn("GLUE", queries[0])

    def test_report_context_gap_items_uses_structured_synthesis_coverage(self):
        question = "What benchmark results demonstrate GLUE performance?"
        context = {
            "synthesis": "The topic is mentioned.",
            "coverage_by_question": [{"question": question, "status": "partial", "source_indexes": [1]}],
        }
        plan = {"objective": "Attention mechanism", "sub_questions": [question]}

        self.assertEqual(synthesis_coverage_gap_questions(context, plan["sub_questions"]), [question])
        self.assertEqual(report_context_gap_items(context, plan), [question])

    def test_synthesis_gap_ignores_stale_missing_when_pack_has_cited_evidence(self):
        question = "What equation is used?"
        context = {
            "coverage_by_question": [{"question": question, "status": "missing", "source_indexes": []}],
            "evidence_packs": [
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [{"source_index": 1, "content": "The equation is present."}],
                }
            ],
        }

        self.assertEqual(synthesis_coverage_gap_questions(context, [question]), [])

    def test_synthesis_gap_ignores_stale_partial_when_per_question_synthesis_has_citations(self):
        question = "What are the major applications of attention mechanisms?"
        context = {
            "coverage_by_question": [{"question": question, "status": "partial", "source_indexes": []}],
            "per_question_synthesis": [
                {
                    "question": question,
                    "synthesis": "Attention is used in image captioning and speech recognition [5] [6].",
                    "source_indexes": [5, 6],
                }
            ],
        }

        self.assertEqual(synthesis_coverage_gap_questions(context, [question]), [])

    def test_resolve_report_coverage_prefers_cited_evidence_pack(self):
        question = "How does the method work?"

        resolved = resolve_report_coverage(
            [{"question": question, "status": "missing", "source_indexes": [], "missing_reason": "No match."}],
            [
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [{"source_index": 2, "content": "The method is explained with source evidence."}],
                }
            ],
            [question],
        )

        self.assertEqual(resolved[0]["status"], "covered")
        self.assertEqual(resolved[0]["source_indexes"], [2])
        self.assertEqual(resolved[0]["missing_reason"], "")

    def test_report_evidence_gap_contradictions_flags_false_gap(self):
        question = "What are the main variants such as Alpha and Beta?"
        report = """
## Main Variants Such As Alpha And Beta
Alpha is covered by source evidence [1].
Beta evidence not provided in the supplied sources.

## References
[1] https://example.com
"""

        false_gaps = report_evidence_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [{"source_index": 1, "content": "Beta is described as a supported variant."}],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [question])

    def test_report_evidence_gap_contradictions_detects_covered_source_denial(self):
        question = "What are the core equations governing the original Bahdanau additive attention mechanism?"
        report = """
## Core Equations Governing Bahdanau Additive Attention
Evidence Gap: The provided sources mention Bahdanau attention but do not contain the explicit compatibility function.
"""

        false_gaps = report_evidence_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "coverage": "covered",
                    "chunks": [
                        {
                            "source_index": 7,
                            "content": "The additive attention compatibility function is available in this source.",
                        }
                    ],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [question])

    def test_report_evidence_gap_contradictions_detects_partial_formula_denial(self):
        question = "What are the core equations of the original additive (Bahdanau) attention mechanism?"
        report = """
## Core Equations Of The Original Additive Bahdanau Attention Mechanism
The supplied evidence does not contain the explicit additive-attention formulas from the original Bahdanau paper [2].

## References
[2] https://arxiv.org/pdf/1409.0473
"""

        false_gaps = report_evidence_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "coverage": "partial",
                    "chunks": [
                        {
                            "source_index": 2,
                            "content": "The alignment model uses a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj), where Wa and Ua are matrices.",
                        }
                    ],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [question])

    def test_report_evidence_gap_contradictions_allows_cited_partial_pack_gap(self):
        question = "What are the standard implementations and APIs for attention in PyTorch and TensorFlow?"
        report = """
## Standard Implementations And APIs For Attention In PyTorch And TensorFlow
TensorFlow provides an online API reference for public functions, classes, and modules [4].
Concrete PyTorch API references are missing from the supplied evidence [4].
"""

        false_gaps = report_evidence_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "coverage": "partial",
                    "missing_facets": ["PyTorch"],
                    "chunks": [
                        {
                            "source_index": 4,
                            "content": "TensorFlow provides API documentation, but the retrieved chunk does not list PyTorch APIs.",
                        }
                    ],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [])

    def test_report_pack_citation_gaps_flags_uncited_matching_section(self):
        question = "What are the core equations of the original additive (Bahdanau) attention mechanism?"
        report = """
## Core Equations Of The Original Additive Bahdanau Attention Mechanism
Bahdanau additive attention uses an alignment score with a tanh compatibility function.

## References
[3] https://example.com/other
"""

        gaps = report_pack_citation_gaps(
            report,
            [
                {
                    "question": question,
                    "coverage": "partial",
                    "chunks": [
                        {
                            "source_index": 2,
                            "content": "The alignment model uses a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj).",
                        }
                    ],
                }
            ],
            [question],
        )

        self.assertEqual(gaps, [question])

    def test_report_pack_citation_gaps_allows_pack_citation(self):
        question = "What are the core equations of the original additive (Bahdanau) attention mechanism?"
        report = """
## Core Equations Of The Original Additive Bahdanau Attention Mechanism
Bahdanau additive attention uses an alignment score with a tanh compatibility function [2].

## References
[2] https://arxiv.org/pdf/1409.0473
"""

        gaps = report_pack_citation_gaps(
            report,
            [
                {
                    "question": question,
                    "coverage": "partial",
                    "chunks": [
                        {
                            "source_index": 2,
                            "content": "The alignment model uses a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj).",
                        }
                    ],
                }
            ],
            [question],
        )

        self.assertEqual(gaps, [])

    def test_report_synthesis_gap_contradictions_flags_false_application_gap(self):
        question = "What are the major applications of attention mechanisms across NLP, vision, and speech?"
        report = """
## Major Applications Of Attention Mechanisms Across NLP Vision And Speech
Attention is used in translation [1].
Evidence for vision and speech applications is missing.
"""

        false_gaps = report_synthesis_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "synthesis": "Vision applications include image captioning [5]. Speech recognition is also supported [6].",
                    "source_indexes": [5, 6],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [question])

    def test_report_synthesis_gap_contradictions_allows_specific_missing_detail(self):
        question = "What are the standard implementations and APIs for attention in PyTorch and TensorFlow?"
        report = """
## Standard Implementations And APIs For Attention In PyTorch And TensorFlow
TensorFlow provides an online API reference for public functions, classes, and modules [4].
Specific PyTorch API references are missing, and exact TensorFlow attention classes are not listed [4].
"""

        false_gaps = report_synthesis_gap_contradictions(
            report,
            [
                {
                    "question": question,
                    "synthesis": (
                        "TensorFlow provides an online API reference for public functions, classes, and modules [4]. "
                        "The exact TensorFlow attention classes are not listed in the retrieved evidence. "
                        "Any description of PyTorch attention APIs is absent."
                    ),
                    "source_indexes": [4],
                }
            ],
            [question],
        )

        self.assertEqual(false_gaps, [])

    def test_report_per_question_synthesis_citation_gaps_flags_dropped_synthesis_sources(self):
        question = "What is the definition of attention?"
        report = """
## Definition Of Attention
Attention lets a model focus on specific input elements.
"""

        gaps = report_per_question_synthesis_citation_gaps(
            report,
            [
                {
                    "question": question,
                    "synthesis": "Attention lets a model focus on specific input elements [5].",
                    "source_indexes": [5],
                }
            ],
            [question],
        )

        self.assertEqual(gaps, [question])

    def test_report_needs_revision_for_false_evidence_gap(self):
        self.assertTrue(
            report_needs_revision(
                {
                    "report_issues": ["report marks covered evidence as a gap: What is covered?"],
                    "schema_issues": [],
                    "synthesis_gaps": [],
                    "false_gap_questions": ["What is covered?"],
                    "coverage": {"missing": []},
                }
            )
        )

    def test_report_revision_feedback_includes_false_gap_questions(self):
        question = "What are the core equations governing Bahdanau attention?"

        feedback = format_report_revision_feedback(
            {
                "report_issues": [],
                "schema_issues": [],
                "coverage": {"missing": []},
                "synthesis_gaps": [],
                "false_gap_questions": [question],
            }
        )

        self.assertIn("false evidence gap to remove", feedback)
        self.assertIn(question, feedback)

    def test_report_revision_feedback_includes_pack_citation_gaps(self):
        question = "What source-backed topic needs a citation?"

        feedback = format_report_revision_feedback(
            {
                "report_issues": [],
                "schema_issues": [],
                "coverage": {"missing": []},
                "synthesis_gaps": [],
                "false_gap_questions": [],
                "pack_citation_gap_questions": [question],
            }
        )

        self.assertIn("missing evidence-pack citation", feedback)
        self.assertIn(question, feedback)

    def test_apply_report_evidence_pack_repairs_removes_false_gap_and_adds_cited_note(self):
        question = "What are the core equations of the original additive attention mechanism?"
        report = """
## Core Equations Of The Original Additive Attention Mechanism
Evidence Gap: The supplied evidence does not contain the explicit formula.

## References
[2] https://arxiv.org/pdf/1409.0473
"""
        packs = [
            {
                "question": question,
                "coverage": "partial",
                "chunks": [
                    {
                        "source_index": 2,
                        "title": "Original paper",
                        "url": "https://arxiv.org/pdf/1409.0473",
                        "content": "The alignment model uses a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj).",
                    }
                ],
            }
        ]

        repaired, repairs = apply_report_evidence_pack_repairs(
            report,
            packs,
            {"false_gap_questions": [question], "pack_citation_gap_questions": []},
            [question],
            [{"index": 2, "url": "https://arxiv.org/pdf/1409.0473"}],
        )

        self.assertEqual(repairs, [question])
        self.assertNotIn("Evidence Gap", repaired)
        self.assertIn("Core evidence", repaired)
        self.assertIn("[2]", repaired)
        self.assertIn("1409.0473", repaired)

    def test_apply_report_evidence_pack_repairs_adds_missing_pack_citation(self):
        question = "What are the core equations of the original additive attention mechanism?"
        report = """
## Core Equations Of The Original Additive Attention Mechanism
Additive attention uses a tanh compatibility function.

## References
No cited source markers were used.
"""
        packs = [
            {
                "question": question,
                "coverage": "partial",
                "chunks": [
                    {
                        "source_index": 2,
                        "title": "Original paper",
                        "url": "https://arxiv.org/pdf/1409.0473",
                        "content": "The alignment model uses a(si-1,hj) = va^T tanh(Wa si-1 + Ua hj).",
                    }
                ],
            }
        ]

        repaired, repairs = apply_report_evidence_pack_repairs(
            report,
            packs,
            {"false_gap_questions": [], "pack_citation_gap_questions": [question]},
            [question],
            [{"index": 2, "url": "https://arxiv.org/pdf/1409.0473"}],
        )

        self.assertEqual(repairs, [question])
        self.assertIn("Core evidence", repaired)
        self.assertIn("[2] https://arxiv.org/pdf/1409.0473", repaired)

    def test_apply_report_repairs_prefer_per_question_synthesis_over_raw_pack_chunk(self):
        question = "What are the major applications of attention mechanisms?"
        report = """
## Major Applications Of Attention Mechanisms
Evidence for vision and speech applications is missing.

## References
No cited source markers were used.
"""
        packs = [
            {
                "question": question,
                "coverage": "covered",
                "chunks": [
                    {
                        "source_index": 2,
                        "title": "PDF",
                        "content": "ut. The quadratic dependence on n poses a challenge for very long input sequences.",
                    }
                ],
            }
        ]
        synthesis = [
            {
                "question": question,
                "synthesis": "Attention is used for image captioning [5] and speech recognition [6].",
                "source_indexes": [5, 6],
            }
        ]

        repaired, repairs = apply_report_evidence_pack_repairs(
            report,
            packs,
            {"false_gap_questions": [question], "pack_citation_gap_questions": []},
            [question],
            [{"index": 5, "url": "https://vision.example"}, {"index": 6, "url": "https://speech.example"}],
            per_question_synthesis=synthesis,
        )

        self.assertEqual(repairs, [question])
        self.assertIn("Per-question synthesis support", repaired)
        self.assertIn("image captioning [5]", repaired)
        self.assertIn("speech recognition [6]", repaired)
        self.assertNotIn("quadratic dependence", repaired)

    def test_per_question_synthesis_repair_note_does_not_cut_mid_sentence(self):
        text = (
            "Attention is used for image captioning [5]. "
            "Speech recognition is supported by the retrieved evidence [6]. "
            "This final sentence should be dropped rather than clipped in the middle of a word [6]."
        )

        snippet = compact_markdown_at_sentence(text, 95)

        self.assertEqual(snippet, "Attention is used for image captioning [5].")

    def test_format_single_question_synthesis_lists_source_markers(self):
        text = format_single_question_synthesis(
            "What is attention?",
            {"synthesis": "Attention focuses input elements [5].", "source_indexes": [5]},
        )

        self.assertIn("Synthesis source markers: [5]", text)
        self.assertIn("Attention focuses input elements [5].", text)

    def test_repair_report_by_sections_regenerates_only_target_topics_and_framing(self):
        question = "What benchmark result is reported?"
        calls = []

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

        def fake_completion(client, **kwargs):
            prompt = kwargs["messages"][1]["content"]
            calls.append(prompt)
            if "Sub-question this section must answer:" in prompt:
                return Response("The benchmark result is 28.4 BLEU, which directly answers the planner question with cited evidence [2].")
            return Response("Updated framing uses the repaired benchmark result [2].")

        original = report_agent_module.create_chat_completion_with_retries
        original_single = report_agent_module.generate_single_report
        report_agent_module.create_chat_completion_with_retries = fake_completion
        report_agent_module.generate_single_report = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-shot repair should not run"))
        try:
            repaired, _, diagnostics = repair_report_by_sections(
                object(),
                "test-model",
                report="""
## 1. Executive Summary
Old summary.

## 2. Introduction and Context
Old intro.

## 3. Topic Sections

### 3.1 Benchmark Result Is Reported
The benchmark result is discussed without the expected citation.

## 4. Cross-cutting Analysis and Synthesis
Old synthesis.

## 5. Limitations and Open Questions
Old limitations.

## 6. Conclusion
Old conclusion.
""",
                objective="Attention mechanism",
                coverage_questions=[question],
                evidence_packs=[
                    {
                        "question": question,
                        "coverage": "covered",
                        "chunks": [{"source_index": 2, "title": "Paper", "content": "The benchmark result is 28.4 BLEU."}],
                    }
                ],
                sources=[{"index": 2, "url": "https://example.com/paper"}],
                validation={"pack_citation_gap_questions": [question], "false_gap_questions": [], "coverage": {"missing": []}, "schema_issues": []},
                repair_feedback="- missing evidence-pack citation to add in matching section: What benchmark result is reported?",
                per_question_synthesis=[],
            )
        finally:
            report_agent_module.create_chat_completion_with_retries = original
            report_agent_module.generate_single_report = original_single

        self.assertEqual(len(calls), 6)
        self.assertEqual(diagnostics["section_repairs"][0]["question"], question)
        self.assertTrue(diagnostics["framing_refreshed"])
        self.assertIn("The benchmark result is 28.4 BLEU", repaired)
        self.assertIn("Updated framing uses the repaired benchmark result [2].", repaired)

    def test_rewrite_missing_sub_question_queries_keeps_focus(self):
        queries = rewrite_missing_sub_question_queries(
            "Attention mechanism",
            ["What are the applications and benefits?"],
        )

        self.assertEqual(len(queries), 1)
        self.assertIn("applications", queries[0])

    def test_remove_unavailable_citation_markers(self):
        cleaned = remove_unavailable_citation_markers("Keep [1], drop [9].", {1})

        self.assertEqual(cleaned, "Keep [1], drop .")

    def test_report_generation_token_cap_stays_under_budget(self):
        self.assertLessEqual(report_generation_token_cap(), DEFAULT_REPORT_TOTAL_TOKEN_BUDGET)

    def test_slugify_filename(self):
        self.assertEqual(slugify_filename("What is Attention Mechanism?"), "what-is-attention-mechanism")


if __name__ == "__main__":
    unittest.main()

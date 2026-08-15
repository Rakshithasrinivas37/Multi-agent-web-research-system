import unittest

from src.agents.report_agent import (
    DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
    clean_markdown,
    dedupe_sources,
    format_report_section_outline,
    format_supporting_evidence,
    missing_evidence_constraints,
    missing_sub_question_coverage,
    normalize_final_report,
    normalize_markdown_headings,
    remove_unavailable_citation_markers,
    report_context_gap_items,
    report_context_gap_queries,
    report_generation_token_cap,
    report_quality_issues,
    report_schema_issues,
    report_self_critique,
    report_sub_question_coverage_check,
    rewrite_missing_sub_question_queries,
    slugify_filename,
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

    def test_format_report_section_outline_uses_topic_headings(self):
        outline = format_report_section_outline(
            [
                "What benchmark results demonstrate GLUE and WMT performance?",
                "How is PyTorch MultiheadAttention used?",
            ]
        )

        self.assertIn("## 1. Benchmark Demonstrate GLUE And WMT Performance", outline)
        self.assertIn("## 2. PyTorch MultiheadAttention Used", outline)

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

import unittest

from src.agents.report_agent import (
    DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
    hard_report_issues,
    markdown_completion_issues,
    report_context_gap_items,
    report_context_gap_queries,
    missing_sub_question_coverage,
    normalize_final_report,
    remove_unavailable_citation_markers,
    report_generation_token_cap,
    report_quality_issues,
    rewrite_missing_sub_question_queries,
)


class ReportAgentValidationTests(unittest.TestCase):
    def test_report_quality_flags_unavailable_citations(self):
        report = """# Topic

## Executive Summary
Supported claim [1]. Unsupported marker [7].

## References
[1] https://example.com/one
"""
        sources = [{"index": 1, "url": "https://example.com/one"}]

        issues = report_quality_issues(report, "", sources=sources)

        self.assertIn("report uses unavailable citations: [7]", issues)

    def test_report_quality_flags_missing_reference_entries(self):
        report = """# Topic

## Executive Summary
Supported claim [1].

## References
"""
        sources = [{"index": 1, "url": "https://example.com/one"}]

        issues = report_quality_issues(report, "", sources=sources)

        self.assertIn("report References section is missing cited sources: [1]", issues)

    def test_report_quality_flags_incomplete_sections(self):
        report = """# Topic

## Executive Summary
This section stops with

## References
No cited source markers were used.
"""

        issues = report_quality_issues(report, "")

        self.assertIn("report contains incomplete sections: Executive Summary", issues)

    def test_report_quality_flags_incomplete_h3_sections(self):
        report = """# Topic

### Executive Summary
This h3 section stops with

## References
No cited source markers were used.
"""

        issues = report_quality_issues(report, "")

        self.assertIn("report contains incomplete sections: Executive Summary", issues)

    def test_report_quality_does_not_flag_h2_container_with_h3_content(self):
        report = """# Topic

## Parent Section
This section has two parts:

### Child Section
This subsection is complete.

## References
No cited source markers were used.
"""

        issues = report_quality_issues(report, "")

        self.assertNotIn("report contains incomplete sections: Parent Section", issues)

    def test_report_quality_requires_references_heading(self):
        report = """# Topic

## Executive Summary
This sentence mentions references but has no reference section.
"""

        issues = report_quality_issues(report, "")

        self.assertIn("report must include a References section", issues)

    def test_remove_unavailable_citation_markers_keeps_valid_markers(self):
        text = "Valid source [1], duplicate style 【2】, and unavailable source [9]."

        sanitized = remove_unavailable_citation_markers(text, {1, 2})

        self.assertIn("[1]", sanitized)
        self.assertIn("[2]", sanitized)
        self.assertNotIn("[9]", sanitized)

    def test_markdown_completion_ignores_trailing_horizontal_rule(self):
        section = """## Section
This section is complete.

---
"""

        self.assertEqual(markdown_completion_issues(section), [])

    def test_normalize_final_report_strips_unavailable_citations_before_references(self):
        report = """# Topic

## Executive Summary
Valid claim [1]. Bad copied marker [9].

## References
[9] https://example.com/bad
"""
        sources = [{"index": 1, "url": "https://example.com/one"}]

        normalized = normalize_final_report(report, sources)

        self.assertIn("[1] https://example.com/one", normalized)
        self.assertNotIn("[9]", normalized)

    def test_normalize_final_report_trims_incomplete_section_tail(self):
        report = """# Topic

## Executive Summary
This sentence is complete.
This final generated line stops with

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("This sentence is complete.", normalized)
        self.assertNotIn("This final generated line stops with", normalized)

    def test_normalize_final_report_trims_incomplete_table_tail(self):
        report = """# Topic

## Data Summary
| Field | Value |
|------|---------|
| Complete | Row |
| Broken | Row

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("| Complete | Row |", normalized)
        self.assertNotIn("| Broken | Row", normalized)
        self.assertEqual(report_quality_issues(normalized, ""), [])

    def test_normalize_final_report_removes_empty_sections(self):
        report = """# Topic

## Complete Section
This section is complete.

## Empty Generated Section

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Complete Section", normalized)
        self.assertNotIn("## Empty Generated Section", normalized)
        self.assertEqual(report_quality_issues(normalized, ""), [])

    def test_normalize_final_report_keeps_parent_sections_with_child_content(self):
        report = """# Topic

## Parent Section

### Child Section
This child content is complete.

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Parent Section", normalized)
        self.assertIn("### Child Section", normalized)
        self.assertEqual(report_quality_issues(normalized, ""), [])

    def test_report_generation_token_caps_stay_under_budget(self):
        self.assertLessEqual(
            report_generation_token_cap(),
            DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
        )

    def test_missing_sub_question_coverage_flags_unanswered_questions(self):
        report = "The report defines attention and explains neural network scoring."
        questions = [
            "What is the definition of attention mechanism?",
            "What are the applications and benefits of attention mechanisms?",
        ]

        missing = missing_sub_question_coverage(report, questions)

        self.assertEqual(missing, [questions[1]])

    def test_rewrite_missing_sub_question_queries_keeps_queries_focused(self):
        queries = rewrite_missing_sub_question_queries(
            "What is attention mechanism?",
            ["What are the applications and benefits?"],
        )

        self.assertEqual(len(queries), 1)
        self.assertIn("applications", queries[0])

    def test_report_context_gap_items_detects_missing_synthesis_coverage(self):
        report_context = {
            "synthesis": """# Notes

### What are the equations for scaled dot-product attention?
- Evidence: None in the retrieved set.
- Gaps: Need the scaled dot-product formula.
""",
            "planner_questions": ["What are the equations for scaled dot-product attention?"],
        }
        research_plan = {"objective": "What is attention?", "sub_questions": report_context["planner_questions"]}

        gaps = report_context_gap_items(report_context, research_plan)

        self.assertTrue(any("scaled dot-product" in gap for gap in gaps))

    def test_report_context_gap_queries_rewrites_preflight_gaps(self):
        report_context = {
            "synthesis": "| Requirement | Status | Notes |\n| Formula | Missing | Need exact equation. |",
            "planner_questions": [],
        }
        research_plan = {"objective": "What is attention?"}

        queries = report_context_gap_queries(report_context, research_plan)

        self.assertEqual(len(queries), 1)
        self.assertIn("exact equation", queries[0])

    def test_hard_report_issues_blocks_stale_missing_evidence(self):
        issues = [
            "report may contain stale missing-evidence statements contradicted by supporting evidence",
            "report must include a References section",
        ]

        self.assertEqual(hard_report_issues(issues), issues)

    def test_report_contract_requires_executive_summary_with_evidence(self):
        report = """# Topic

## Details
Supported claim [1].

## References
[1] https://example.com
"""

        issues = report_quality_issues(report, "Evidence: supported claim [1].")

        self.assertIn("report must include an Executive Summary section", issues)

    def test_report_contract_requires_supported_formula_preservation(self):
        report = """# Topic

## Executive Summary
Attention uses weighted values [1].

## References
[1] https://example.com
"""
        evidence = r"Evidence: \[Attention(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V\] [1]."

        issues = report_quality_issues(report, evidence)

        self.assertIn("report omits supported equations or formulas", issues)


if __name__ == "__main__":
    unittest.main()

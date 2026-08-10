import unittest

from src.agents.report_agent import (
    markdown_completion_issues,
    normalize_final_report,
    remove_unavailable_citation_markers,
    report_quality_issues,
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


if __name__ == "__main__":
    unittest.main()

import unittest

from src.agents.report_agent import report_quality_issues


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


if __name__ == "__main__":
    unittest.main()

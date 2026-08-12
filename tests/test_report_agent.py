import unittest

from src.agents.report_agent import (
    build_report_prompt,
    dedupe_sources_by_url,
    format_question_evidence_packet,
    format_supporting_evidence,
    missing_sub_question_coverage,
    normalize_final_report,
    remove_unavailable_citation_markers,
    report_context_gap_items,
    report_context_gap_queries,
    report_generation_token_cap,
    report_quality_issues,
    rewrite_missing_sub_question_queries,
    slugify_filename,
)


class ReportAgentTests(unittest.TestCase):
    def test_prompt_requires_one_section_per_planner_question(self):
        prompt = build_report_prompt(
            objective="Attention mechanism",
            output_format="deep_dive",
            planner_questions=["What is attention?", "How is it implemented?"],
            sources=[{"index": 1, "url": "https://example.com", "title": "Example"}],
            question_packet="Q1 evidence",
            evidence_text="Evidence",
            synthesis="Synthesis",
        )

        self.assertIn('Include "## Executive Summary"', prompt)
        self.assertIn('exactly one "##" section for every planner sub-question', prompt)
        self.assertIn("## 1. <question>", prompt)
        self.assertIn("Do not concentrate only on equations", prompt)

    def test_format_question_evidence_packet_groups_evidence_by_question(self):
        packet = format_question_evidence_packet(
            ["What ImageNet results are reported for Vision Transformers?"],
            synthesis="ViT uses image patches.",
            evidence_text="[1] Vision Transformer reports ImageNet accuracy after pretraining.",
            sources=[{"index": 1, "url": "https://arxiv.org/pdf/2010.11929", "title": "ViT"}],
        )

        self.assertIn("Q1. What ImageNet results", packet)
        self.assertIn("Coverage: Covered", packet)
        self.assertIn("ImageNet accuracy", packet)

    def test_missing_sub_question_coverage_flags_missing_and_short_sections(self):
        questions = [
            "What is attention?",
            "What are computational complexity and memory trade-offs?",
        ]
        report = """# Topic

## Executive Summary
Overview.

## 1. What is attention?
Attention selects relevant information from available context. It assigns weights to inputs, combines the weighted values, and lets the model focus on information that matters for a prediction. This section is intentionally detailed enough to satisfy the section-depth gate. It also explains the purpose, the computation at a high level, and why the mechanism helps neural networks preserve useful context across longer inputs.

## 2. What are computational complexity and memory trade-offs?
Too short.

## References
No cited source markers were used.
"""

        missing = missing_sub_question_coverage(report, questions)

        self.assertEqual(missing, [questions[1]])

    def test_missing_sub_question_coverage_requires_numbered_sections(self):
        questions = ["What are applications in NLP?"]
        report = """# Topic

## Executive Summary
Overview.

## Applications
NLP applications include translation and language modeling. This has enough words, but it is not under the required numbered planner-question heading, so the gate should still reject it.

## References
No cited source markers were used.
"""

        self.assertEqual(missing_sub_question_coverage(report, questions), questions)

    def test_report_quality_flags_unavailable_citations_and_missing_refs(self):
        report = """# Topic

## Executive Summary
Supported claim [1]. Unsupported claim [9].

## 1. What is attention?
Attention selects relevant information from context and uses weighted combinations of values to produce contextual representations. The section contains enough detail to be considered complete.

## References
"""
        issues = report_quality_issues(report, "Evidence", sources=[{"index": 1, "url": "https://example.com"}])

        self.assertIn("report uses unavailable citations: [9]", issues)
        self.assertIn("report References section is missing cited sources: [1]", issues)

    def test_normalize_final_report_rebuilds_references_and_removes_bad_markers(self):
        report = """# Topic

## Executive Summary
Supported claim [1]. Bad marker [7].

## 1. What is attention?
Attention selects relevant context and combines values according to learned weights. It supports detailed contextual reasoning in sequence and multimodal models with evidence-backed citations [1].

## References
[7] https://bad.example
"""

        normalized = normalize_final_report(report, [{"index": 1, "url": "https://example.com/one"}])

        self.assertIn("[1] https://example.com/one", normalized)
        self.assertNotIn("[7]", normalized)

    def test_remove_unavailable_citation_markers_normalizes_marker_style(self):
        text = "Valid 【1】, also valid [2], invalid [9]."

        cleaned = remove_unavailable_citation_markers(text, {1, 2})

        self.assertIn("[1]", cleaned)
        self.assertIn("[2]", cleaned)
        self.assertNotIn("[9]", cleaned)

    def test_dedupe_sources_by_url_maps_duplicate_indexes(self):
        sources, aliases = dedupe_sources_by_url(
            [
                {"index": 1, "url": "https://example.com/a", "title": "A"},
                {"index": 5, "url": "https://example.com/a", "title": "Duplicate"},
            ]
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(aliases[5], 1)

    def test_format_supporting_evidence_uses_citation_aliases(self):
        context = {
            "supporting_chunks": [
                {
                    "source_index": 5,
                    "title": "Duplicate source",
                    "url": "https://example.com/a",
                    "content": "Attention evidence.",
                }
            ]
        }

        evidence = format_supporting_evidence(context, {5: 1})

        self.assertIn("[1]", evidence)
        self.assertIn("Attention evidence", evidence)

    def test_report_context_gap_items_uses_chunks_not_only_synthesis(self):
        context = {
            "synthesis": "The synthesis says this is missing.",
            "planner_questions": ["What ImageNet results are reported?"],
            "sources": [{"index": 1, "url": "https://arxiv.org/pdf/2010.11929"}],
            "supporting_chunks": [
                {
                    "source_index": 1,
                    "url": "https://arxiv.org/pdf/2010.11929",
                    "content": "Vision Transformer reports ImageNet results and accuracy.",
                }
            ],
        }
        plan = {"objective": "Attention mechanism", "sub_questions": context["planner_questions"]}

        self.assertEqual(report_context_gap_items(context, plan), [])

    def test_report_context_gap_queries_rewrites_missing_questions(self):
        context = {"synthesis": "", "planner_questions": ["What is the PyTorch API?"]}
        plan = {"objective": "Attention mechanism", "sub_questions": context["planner_questions"]}

        queries = report_context_gap_queries(context, plan)

        self.assertEqual(queries, ["Attention mechanism What is the PyTorch API?"])

    def test_rewrite_missing_sub_question_queries_and_slug(self):
        self.assertEqual(
            rewrite_missing_sub_question_queries("Attention", ["What is self-attention?"]),
            ["Attention What is self-attention?"],
        )
        self.assertEqual(slugify_filename("What is Attention Mechanism?"), "what-is-attention-mechanism")

    def test_report_generation_token_cap_stays_under_budget(self):
        self.assertLessEqual(report_generation_token_cap(), 10000)


if __name__ == "__main__":
    unittest.main()

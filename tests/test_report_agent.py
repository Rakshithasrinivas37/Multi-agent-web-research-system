import unittest

from src.agents.report_agent import (
    DEFAULT_REPORT_TOTAL_TOKEN_BUDGET,
    ensure_supported_api_details,
    format_source_priority_guidance,
    hard_report_issues,
    markdown_completion_issues,
    format_evidence_coverage_brief,
    format_memory_signal_evidence,
    format_planner_evidence_packet,
    missing_evidence_constraints,
    report_context_gap_items,
    report_context_gap_queries,
    missing_sub_question_coverage,
    normalize_final_report,
    normalize_report_for_validation,
    remove_conflicting_missing_evidence_statements,
    remove_placeholder_citations,
    remove_unavailable_citation_markers,
    report_generation_token_cap,
    report_quality_issues,
    rewrite_missing_sub_question_queries,
    clean_markdown,
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

    def test_clean_markdown_strips_leaked_thinking_blocks(self):
        text = "Useful synthesis.\n<think>private reasoning with missing gap noise</think>\nFinal evidence."

        cleaned = clean_markdown(text)

        self.assertIn("Useful synthesis.", cleaned)
        self.assertIn("Final evidence.", cleaned)
        self.assertNotIn("private reasoning", cleaned)

    def test_clean_markdown_strips_unclosed_thinking_blocks(self):
        text = "Useful synthesis.\n<think>private reasoning with stale gap notes"

        cleaned = clean_markdown(text)

        self.assertEqual(cleaned, "Useful synthesis.")

    def test_missing_evidence_constraints_ignores_covered_table_rows(self):
        synthesis = """| Requirement | Status | Notes |
|---|---|---|
| Formula | Covered | No missing evidence remains. |
| Benchmark | Partial | Exact scores are missing. |
"""

        constraints = missing_evidence_constraints(synthesis)

        self.assertEqual(constraints, ["Benchmark: Exact scores are missing."])

    def test_format_planner_evidence_packet_marks_supported_questions_covered(self):
        report_context = {
            "supporting_chunks": [
                {
                    "source_index": 7,
                    "url": "https://arxiv.org/pdf/1409.0473",
                    "title": "Bahdanau attention",
                    "content": "The context vector c_i is computed as a weighted sum c_i = sum_j alpha_ij h_j, where alpha_ij is computed with softmax from the alignment model e_ij.",
                }
            ]
        }
        sources = [{"index": 7, "url": "https://arxiv.org/pdf/1409.0473", "title": "Bahdanau attention"}]

        packet = format_planner_evidence_packet(
            report_context,
            ["What are the equations behind Bahdanau additive attention?"],
            sources,
        )

        self.assertIn("Status: Covered", packet)
        self.assertIn("Allowed Evidence Gaps: none", packet)
        self.assertIn("[7]", packet)

    def test_format_planner_evidence_packet_marks_missing_subtopics(self):
        report_context = {
            "supporting_chunks": [
                {
                    "source_index": 8,
                    "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
                    "title": "PyTorch MultiheadAttention",
                    "content": "torch.nn.MultiheadAttention implements multi-head attention with embed_dim and num_heads.",
                }
            ]
        }
        sources = [
            {
                "index": 8,
                "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
                "title": "PyTorch MultiheadAttention",
            }
        ]

        packet = format_planner_evidence_packet(
            report_context,
            ["What are the standard implementations and APIs for attention in PyTorch and TensorFlow?"],
            sources,
        )

        self.assertIn("Status: Partial", packet)
        self.assertIn("Covered topics: pytorch", packet)
        self.assertIn("Allowed Evidence Gaps: tensorflow", packet)

    def test_normalize_report_for_validation_removes_stale_evidence_status_when_packet_covers_topic(self):
        report = """# Topic

## Executive Summary
Bahdanau attention computes a context vector with softmax-normalized alignment weights.

## Bahdanau Attention
The context vector is computed as \\(c_i = \\sum_j \\alpha_{ij}h_j\\).
Evidence status: the provided sources do not include the explicit equations.

## References
[7] https://arxiv.org/pdf/1409.0473
"""
        sources = [{"index": 7, "url": "https://arxiv.org/pdf/1409.0473"}]
        evidence = """Planner evidence packet:
Question: What are the equations behind Bahdanau additive attention?
Status: Covered
Best evidence:
- [7] The context vector c_i is computed as a weighted sum c_i = sum_j alpha_ij h_j, with alpha_ij computed by softmax from e_ij.
Allowed Evidence Gaps: none for this question
"""

        normalized = normalize_report_for_validation(report, sources, evidence)

        self.assertIn("context vector", normalized)
        self.assertNotIn("do not include the explicit equations", normalized)

    def test_report_quality_flags_incomplete_sections(self):
        report = """# Topic

## Executive Summary
This section stops with

## References
No cited source markers were used.
"""

        issues = report_quality_issues(report, "")

        self.assertIn("report contains incomplete sections: Executive Summary", issues)

    def test_report_quality_flags_placeholder_citations(self):
        report = """# Topic

## Executive Summary
Luong attention uses multiplicative scoring [uncited].

## References
No cited source markers were used.
"""

        issues = report_quality_issues(report, "")

        self.assertIn("report contains placeholder citation markers", issues)

    def test_report_quality_flags_named_topics_absent_from_evidence(self):
        report = """# Topic

## Executive Summary
Longformer uses sparse attention for long documents.
Longformer reduces memory for long inputs.

## References
No cited source markers were used.
"""
        evidence = "Evidence discusses Transformer and Linformer only."

        issues = report_quality_issues(report, evidence)

        self.assertIn("report mentions unsupported named topics: longformer", issues)

    def test_remove_placeholder_citations_strips_uncited_markers(self):
        report = "The equation appears in the uncited excerpt [uncited]."

        cleaned = remove_placeholder_citations(report)

        self.assertNotIn("[uncited]", cleaned)
        self.assertNotIn("uncited excerpt", cleaned.lower())

    def test_normalize_report_removes_unsupported_named_topic_rows(self):
        report = """# Topic

## Executive Summary
Linformer is supported.

## Variants
| Variant | Detail |
|---|---|
| Linformer | Low-rank attention. |
| Longformer | Sliding-window attention. |

## References
No cited source markers were used.
"""
        evidence = "Linformer uses low-rank attention."

        normalized = normalize_report_for_validation(report, [], evidence)

        self.assertIn("Linformer", normalized)
        self.assertNotIn("Longformer", normalized)

    def test_normalize_report_removes_unsupported_inference_lines(self):
        report = """# Topic

## Executive Summary
Vision models are discussed.
The original paper is not listed but inferred from the repository.

## References
No cited source markers were used.
"""

        normalized = normalize_report_for_validation(report, [], "Vision models are discussed.")

        self.assertIn("Vision models are discussed.", normalized)
        self.assertNotIn("inferred", normalized.lower())

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

    def test_normalize_final_report_splits_combined_citations(self):
        report = """# Topic

## Executive Summary
Supported implementation details use two sources [2, 26].

## References
[2] https://example.com/two
[26] https://example.com/twenty-six
"""
        sources = [
            {"index": 2, "url": "https://example.com/two"},
            {"index": 26, "url": "https://example.com/twenty-six"},
        ]

        normalized = normalize_final_report(report, sources)

        self.assertIn("[2] [26]", normalized)
        self.assertIn("[2] https://example.com/two", normalized)
        self.assertIn("[26] https://example.com/twenty-six", normalized)
        self.assertNotIn("[2, 26]", normalized)

    def test_normalize_final_report_removes_evidence_gap_placeholders(self):
        report = """# Topic

## Executive Summary
Vision Transformers use image patches [— Evidence Gap —].

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("Vision Transformers use image patches.", normalized)
        self.assertNotIn("Evidence Gap", normalized)

    def test_normalize_final_report_rewrites_topic_citation_to_matching_source(self):
        report = """# Topic

## Executive Summary
Linformer reduces self-attention complexity [2].

## References
[2] https://arxiv.org/pdf/2010.11929
"""
        sources = [
            {"index": 2, "url": "https://arxiv.org/pdf/2010.11929", "title": "ViT"},
            {"index": 22, "url": "https://arxiv.org/pdf/2006.04768", "title": "Linformer"},
        ]

        normalized = normalize_final_report(report, sources)

        self.assertIn("Linformer reduces self-attention complexity [22].", normalized)
        self.assertIn("[22] https://arxiv.org/pdf/2006.04768", normalized)
        self.assertNotIn("[2] https://arxiv.org/pdf/2010.11929", normalized)

    def test_normalize_final_report_removes_unsupported_topic_citation_line(self):
        report = """# Topic

## Executive Summary
Attention has applications in many modalities.
Conformer combines convolution and self-attention for speech [8].

## References
[8] https://arxiv.org/pdf/2009.06732
"""
        sources = [{"index": 8, "url": "https://arxiv.org/pdf/2009.06732", "title": "Efficient Transformers"}]

        normalized = normalize_final_report(report, sources)

        self.assertIn("Attention has applications in many modalities.", normalized)
        self.assertNotIn("Conformer", normalized)
        self.assertNotIn("[8]", normalized)

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

    def test_normalize_final_report_removes_malformed_table_rows(self):
        report = """# Topic

## Executive Summary
This section is complete.

| Component | Role |
|---|---|
| Encoder | Reads input. |
|
|

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertNotIn("\n|\n", normalized)
        self.assertIn("| Encoder | Reads input. |", normalized)

    def test_normalize_final_report_removes_placeholder_table_rows(self):
        report = """# Topic

## Executive Summary
This section is complete.

| Model | Dataset | Score |
|---|---|---|
| Transformer | WMT 2014 | 28.4 BLEU |
| — | — | — |

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("| Transformer | WMT 2014 | 28.4 BLEU |", normalized)
        self.assertNotIn("| — | — | — |", normalized)

    def test_normalize_final_report_removes_empty_math_only_section(self):
        report = """# Topic

## Executive Summary
This section is complete.

## Bahdanau Attention
Supported content remains.

### Architecture Evidence Gap
\\[
\\]

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Bahdanau Attention", normalized)
        self.assertNotIn("Architecture Evidence Gap", normalized)
        self.assertNotIn("\\[\n\\]", normalized)

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

    def test_normalize_final_report_removes_still_incomplete_sections(self):
        report = """# Topic

## Executive Summary
This section is complete.

## TensorFlow/Keras Implementation
The implementation details start here and

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertNotIn("TensorFlow/Keras Implementation", normalized)
        self.assertEqual(report_quality_issues(normalized, ""), [])

    def test_normalize_final_report_moves_summary_to_executive_summary(self):
        report = """**Deep Dive**

### 1. Details
This section is complete.

### 10. Summary
This is the generated summary.

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Executive Summary", normalized)
        self.assertIn("This is the generated summary.", normalized)
        self.assertNotIn("### 10. Summary", normalized)

    def test_normalize_final_report_inserts_executive_summary_when_missing(self):
        report = """# Topic

## Details
This section explains the available evidence.

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Executive Summary", normalized)
        self.assertIn("This section explains the available evidence.", normalized)

    def test_normalize_final_report_removes_boilerplate_evidence_gaps(self):
        report = """# Topic

## Executive Summary
This report covers the available evidence.

## Evidence Gaps
These gaps are acknowledged to avoid overstating unsupported details.

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertNotIn("## Evidence Gaps", normalized)

    def test_normalize_final_report_keeps_substantive_evidence_gaps(self):
        report = """# Topic

## Executive Summary
This report covers the available evidence.

## Evidence Gaps
- Exact benchmark scores were not present in the supplied evidence.

## References
No cited source markers were used.
"""

        normalized = normalize_final_report(report, [])

        self.assertIn("## Evidence Gaps", normalized)
        self.assertIn("Exact benchmark scores", normalized)

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

    def test_format_memory_signal_evidence_adds_browser_benchmark_signal(self):
        report_context = {
            "browser_results": [
                {
                    "sources": [
                        {
                            "url": "https://arxiv.org/abs/1706.03762",
                            "full_content": (
                                "The model achieves 28.4 BLEU on WMT 2014 English-to-German "
                                "and 41.8 BLEU on English-to-French."
                            ),
                        }
                    ]
                }
            ]
        }
        sources = [{"index": 1, "url": "https://arxiv.org/pdf/1706.03762"}]

        evidence = format_memory_signal_evidence(report_context, sources, {}, "No selected benchmark chunk.")

        self.assertIn("[1]", evidence)
        self.assertIn("28.4 BLEU", evidence)

    def test_format_source_priority_guidance_prefers_primary_sources(self):
        sources = [
            {"index": 1, "title": "Attention Is All You Need", "url": "https://arxiv.org/pdf/1706.03762"},
            {"index": 2, "title": "Background Article", "url": "https://example.com/attention"},
        ]

        guidance = format_source_priority_guidance(sources)

        self.assertIn("Primary/official sources: [1] Attention Is All You Need", guidance)
        self.assertIn("background sources", guidance)
        self.assertIn("[2] Background Article", guidance)

    def test_format_evidence_coverage_brief_separates_supported_and_missing_parts(self):
        evidence = "Evidence: Linformer uses a low-rank approximation and reduces complexity from O(n2) to O(n)."
        missing = "- Efficient variants: Linformer is partial, Performer details are not present."

        brief = format_evidence_coverage_brief(
            ["What efficient attention variants are covered?"],
            "",
            evidence,
            missing,
        )

        self.assertIn("Supported evidence signals", brief)
        self.assertIn("low-rank", brief)
        self.assertIn("Performer", brief)

    def test_hard_report_issues_blocks_stale_missing_evidence(self):
        issues = [
            "report may contain stale missing-evidence statements contradicted by supporting evidence",
            "report must include a References section",
        ]

        self.assertEqual(hard_report_issues(issues), issues)

    def test_remove_conflicting_missing_evidence_statements_drops_exact_missing_sentence(self):
        report = r"""# Topic

## Executive Summary
The formula \(Attention(Q,K,V)=softmax(QK^T)V\) is missing from the evidence. Keep this sentence.

## References
No cited source markers were used.
"""
        evidence = r"Evidence includes \(Attention(Q,K,V)=softmax(QK^T)V\)."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertNotIn("is missing from the evidence", cleaned)
        self.assertIn("Keep this sentence.", cleaned)

    def test_remove_conflicting_missing_evidence_statements_keeps_unresolved_gap(self):
        report = """# Topic

## Executive Summary
TensorFlow API details are not present in the evidence.

## References
No cited source markers were used.
"""
        evidence = "Evidence discusses PyTorch only."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertIn("TensorFlow API details are not present", cleaned)

    def test_remove_conflicting_missing_evidence_statements_drops_table_gap_row(self):
        report = """# Topic

## Executive Summary
Supported benchmark details are available.

| Topic | Status | Notes |
|---|---|---|
| BLEU benchmarks | Missing | No BLEU numbers are present. |
| GLUE benchmarks | Missing | No GLUE scores are present. |

## References
No cited source markers were used.
"""
        evidence = "The Transformer model achieves 28.4 BLEU on WMT 2014 English-to-German."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertNotIn("BLEU benchmarks", cleaned)
        self.assertIn("GLUE benchmarks", cleaned)

    def test_remove_conflicting_missing_evidence_statements_drops_bullet_gap(self):
        report = """# Topic

## Executive Summary
Supported API details are available.

- PyTorch API details are missing.
- TensorFlow API details are missing.

## References
No cited source markers were used.
"""
        evidence = "Supporting evidence includes torch.nn.MultiheadAttention."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertNotIn("PyTorch API details are missing", cleaned)
        self.assertIn("TensorFlow API details are missing", cleaned)

    def test_remove_conflicting_missing_evidence_statements_drops_absent_gap(self):
        report = """# Topic

## Executive Summary
Supported API details are available.

- PyTorch API details are absent.
- TensorFlow API details are absent.

## References
No cited source markers were used.
"""
        evidence = "Supporting evidence includes torch.nn.MultiheadAttention."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertNotIn("PyTorch API details are absent", cleaned)
        self.assertIn("TensorFlow API details are absent", cleaned)

    def test_ensure_supported_api_details_adds_omitted_api_section(self):
        report = """# Topic

## Executive Summary
This report summarizes implementation evidence.

## References
[1] https://docs.example.com
"""
        evidence = "[1] Evidence: torch.nn.MultiheadAttention supports multi-head attention."

        updated = ensure_supported_api_details(report, evidence)

        self.assertIn("## Implementation APIs", updated)
        self.assertIn("`torch.nn.MultiheadAttention`", updated)
        self.assertIn("[1]", updated)

    def test_ensure_supported_api_details_ignores_incidental_torch_helpers(self):
        report = """# Topic

## Executive Summary
This report summarizes evidence.

## References
No cited source markers were used.
"""
        evidence = "Example setup uses torch.manual_seed, torch.nn.Embedding, and torch.Size."

        updated = ensure_supported_api_details(report, evidence)

        self.assertNotIn("## Implementation APIs", updated)

    def test_normalize_report_for_validation_removes_weak_implementation_api_section(self):
        report = """# Topic

## Executive Summary
This report summarizes evidence.

## Implementation APIs
- `torch.manual_seed` is present in the supporting evidence [1].
- `torch.Size` is present in the supporting evidence.

## References
[1] https://example.com
"""
        sources = [{"index": 1, "url": "https://example.com"}]
        evidence = "Example setup uses torch.manual_seed and torch.Size."

        normalized = normalize_report_for_validation(report, sources, evidence)

        self.assertNotIn("## Implementation APIs", normalized)
        self.assertNotIn("torch.manual_seed", normalized)

    def test_normalize_report_for_validation_removes_resolved_evidence_gap_rows(self):
        report = r"""# Topic

## Executive Summary
Scaled dot-product attention uses \(\text{softmax}(QK^{\top}/\sqrt{d_k})V\).
TensorFlow offers `tf.keras.layers.MultiHeadAttention`.

## Evidence Gaps
| Planner Sub-question | Missing / Partial Detail | Reason |
|---|---|---|
| Scaled dot-product formula | Explicit `1/√d_k` scaling term | Only the unscaled formulation appears. |
| TensorFlow API | Signature and argument semantics | No supporting chunk from TensorFlow is present. |
| Statistical testing | Confidence intervals | No evidence includes confidence intervals. |

## References
[1] https://example.com
"""
        evidence = "Evidence includes tf.keras.layers.MultiHeadAttention and the formula softmax(QK^T / sqrt(d_k)) V."
        sources = [{"index": 1, "url": "https://example.com"}]

        normalized = normalize_report_for_validation(report, sources, evidence)

        self.assertNotIn("Scaled dot-product formula | Explicit", normalized)
        self.assertNotIn("TensorFlow API | Signature", normalized)
        self.assertIn("Statistical testing", normalized)

    def test_normalize_report_for_validation_uses_synthesis_covered_rows(self):
        report = r"""# Topic

## Executive Summary
This report summarizes attention evidence.

## Evidence Gaps
| Planner Sub-question | Missing / Partial Detail | Reason |
|---|---|---|
| Scaled dot-product formula | Explicit scaling term | Only the unscaled formulation appears. |
| Vision benchmarks | Accuracy numbers | No supporting vision benchmark evidence is present. |

## References
No cited source markers were used.
"""
        synthesis = """| Requirement | Status | Evidence Source(s) |
|---|---|---|
| Scaled-dot-product equation | Covered | [4] |
| Benchmark impact on vision tasks | Missing | – |
"""

        normalized = normalize_report_for_validation(report, [], "", synthesis=synthesis)

        self.assertNotIn("Scaled dot-product formula", normalized)
        self.assertIn("Vision benchmarks", normalized)

    def test_normalize_report_for_validation_removes_inline_resolved_gap(self):
        report = """# Topic

## Luong Attention
Luong attention uses multiplicative dot-product scoring and global/local alignment.
Evidence Gap: No direct citation of the Luong paper is present in the provided sources.

## References
[1] https://arxiv.org/pdf/1508.04025
"""
        evidence = "[1] Luong paper evidence: multiplicative dot-product scoring and global/local alignment."
        sources = [{"index": 1, "url": "https://arxiv.org/pdf/1508.04025"}]

        normalized = normalize_report_for_validation(report, sources, evidence)

        self.assertIn("Luong attention uses multiplicative", normalized)
        self.assertNotIn("No direct citation of the Luong paper", normalized)

    def test_remove_conflicting_missing_evidence_statements_drops_heading_gap(self):
        report = r"""# Topic

## Executive Summary
Supported detail is available.

### Formula \(Attention(Q,K,V)=softmax(QK^T)V\) is missing

## References
No cited source markers were used.
"""
        evidence = r"Evidence includes \(Attention(Q,K,V)=softmax(QK^T)V\)."

        cleaned = remove_conflicting_missing_evidence_statements(report, evidence)

        self.assertNotIn("Formula", cleaned)
        self.assertNotIn("is missing", cleaned)

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

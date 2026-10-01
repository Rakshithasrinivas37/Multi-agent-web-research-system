"""Contract tests for the report pipeline, using fixed evidence and mocked LLM output."""

import unittest
from unittest.mock import patch

import src.agents.report_agent as report_agent


QUESTION = "What is the mathematical formulation of scaled dot-product attention, including the equation?"


def fixture_context():
    synthesis = (
        "Scaled dot-product attention computes a softmax over query-key dot products divided by the square root "
        "of the key dimension, then weights the values. Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V [1]. **Exact missing details** "
        "The retrieved note does not provide masking variants or runtime measurements."
    )
    return {
        "objective": "Scaled dot-product attention",
        "planner_questions": [QUESTION],
        "per_question_synthesis": [{
            "question": QUESTION,
            "coverage": "partial",
            "source_indexes": [1, 999],
            "synthesis": synthesis,
        }],
        "evidence_packs": [{
            "question": QUESTION,
            "coverage": "partial",
            "chunks": [{"source_index": 2, "content": "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V is the scaled dot-product formulation."}],
        }],
        "sources": [
            {"index": 1, "title": "Transformer paper", "url": "https://arxiv.org/abs/1706.03762"},
            {"index": 2, "title": "Attention reference", "url": "https://example.org/attention"},
        ],
        "coverage_by_question": [{"question": QUESTION, "status": "partial"}],
    }


GOOD_REPORT = r"""## 1. Executive Summary
The evidence provides the scaled dot-product attention formulation [1]. Details on masking variants and runtime remain unavailable.

## 2. Introduction and Context
This report examines the formula and its documented limitations [1].

## 3. Topic Sections
### 3.1. Mathematical Formulation Of Scaled Dot-product Attention Including Equation
Scaled dot-product attention normalizes query-key scores by the square root of the key dimension before weighting values [1].

**Core equation:**
\[\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{d_k})V\]

Masking variants and runtime measurements are not provided in the supplied evidence.

## 4. Cross-cutting Analysis and Synthesis
The formulation connects query-key matching to a weighted combination of values [1].

## 5. Limitations and Open Questions
The supplied evidence does not provide masking variants or runtime measurements.

## 6. Conclusion
The equation is supported, while masking variants and runtime measurements remain unresolved [1].
"""


class ReportEvidenceContractTests(unittest.TestCase):
    def test_contract_separates_supported_gaps_and_question_specific_citations(self):
        context = fixture_context()
        notes = report_agent.per_question_synthesis_by_question(context["per_question_synthesis"])
        packs = {report_agent.normalize_heading(item["question"]): item for item in context["evidence_packs"]}
        contracts = report_agent.build_question_evidence_contracts(
            [QUESTION], notes, packs, context["sources"]
        )
        contract = contracts[report_agent.normalize_heading(QUESTION)]

        self.assertIn("softmax", contract["supported"])
        self.assertIn("runtime measurements", contract["missing_details"])
        self.assertEqual(contract["synthesis_source_indexes"], [1])
        self.assertEqual(contract["pack_source_indexes"], [2])
        self.assertEqual(contract["retrieved_chunks"][0]["source_index"], 2)
        self.assertIn("scaled dot-product formulation", contract["retrieved_chunks"][0]["content"])

    def test_prompt_chunks_are_bounded_source_diverse_and_citable(self):
        context = fixture_context()
        pack = context["evidence_packs"][0]
        pack["chunks"] = [
            {"source_index": 2, "title": "Primary", "content": "Primary evidence on scaled attention."},
            {"source_index": 2, "title": "Primary", "content": "Additional primary evidence on scaled attention."},
            {"source_index": 1, "title": "Paper", "content": "Independent evidence on the equation [77]."},
            {"source_index": 2, "title": "Primary", "content": "Third primary detail."},
            {"source_index": 1, "title": "Paper", "content": "Second independent detail."},
            {"source_index": 2, "title": "Primary", "content": "Fourth primary detail."},
            {"source_index": 999, "title": "Unknown", "content": "Must not appear."},
            {"title": "Uncited", "content": "Must also not appear."},
        ]
        contracts = report_agent.build_question_evidence_contracts(
            [QUESTION],
            report_agent.per_question_synthesis_by_question(context["per_question_synthesis"]),
            {report_agent.normalize_heading(QUESTION): pack},
            context["sources"],
        )
        chunks = contracts[report_agent.normalize_heading(QUESTION)]["retrieved_chunks"]

        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0]["source_index"], 2)
        self.assertEqual(chunks[1]["source_index"], 1)
        self.assertNotIn(999, [chunk["source_index"] for chunk in chunks])
        self.assertNotIn("Must not appear", report_agent.format_contract_chunks(chunks))
        self.assertNotIn("[77]", report_agent.format_contract_chunks(chunks))

    def test_cited_covered_pack_overrides_stale_missing_coverage_row(self):
        context = fixture_context()
        context["evidence_packs"][0]["coverage"] = "covered"
        context["coverage_by_question"][0]["status"] = "missing"
        contracts = report_agent.build_question_evidence_contracts(
            [QUESTION],
            report_agent.per_question_synthesis_by_question(context["per_question_synthesis"]),
            {report_agent.normalize_heading(QUESTION): context["evidence_packs"][0]},
            context["sources"],
            context["coverage_by_question"],
        )

        self.assertEqual(contracts[report_agent.normalize_heading(QUESTION)]["coverage"], "covered")
        self.assertEqual(report_agent.synthesis_coverage_gap_questions(context, [QUESTION]), [])

    def test_end_to_end_report_keeps_contract_boundaries_and_validates(self):
        captured = {}

        def fake_llm(model, prompt):
            captured["prompt"] = prompt
            return GOOD_REPORT, "mocked_llm", ""

        with patch.object(report_agent, "generate_report_with_llm", side_effect=fake_llm):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertIn("Markers cited by this question's synthesis: [1]", captured["prompt"])
        self.assertIn("Markers available in its evidence pack (cite only for claims directly supported there): [2]", captured["prompt"])
        self.assertIn("runtime measurements", captured["prompt"])
        self.assertIn("Retrieved evidence for this question", captured["prompt"])
        self.assertIn("[2] Source 2: Attention(Q,K,V)", captured["prompt"])
        self.assertNotIn("[999]", captured["prompt"])
        self.assertIn("[1] https://arxiv.org/abs/1706.03762", payload["report"])
        self.assertEqual(payload["diagnostics"]["report_schema_issues"], [])
        self.assertEqual(payload["diagnostics"]["report_finalization_status"], "clean_with_evidence_gaps")

    def test_uncited_or_malformed_llm_draft_can_be_replaced_by_better_deterministic_report(self):
        bad_report = GOOD_REPORT.replace(
            "This report examines the formula and its documented limitations [1].",
            "Attention is widely used for sequence tasks without citation.",
        )
        with patch.object(
            report_agent, "generate_report_with_llm",
            return_value=(bad_report, "mocked_llm", ""),
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertTrue(payload["diagnostics"]["report_quality_fallback"]["attempted"])
        self.assertTrue(payload["diagnostics"]["report_quality_fallback"]["accepted"])
        self.assertEqual(payload["diagnostics"]["report_issues"], [])
        self.assertIn("validated_fallback", payload["diagnostics"]["report_generation_mode"])

    def test_llm_failure_returns_deterministic_report_instead_of_raising(self):
        with patch.object(
            report_agent, "generate_report_with_llm",
            return_value=(None, "deterministic_fallback_llm_error", "mock model unavailable"),
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertTrue(payload["report"])
        self.assertIn("## 6. Conclusion", payload["report"])
        self.assertEqual(payload["diagnostics"]["report_generation_error"], "mock model unavailable")


if __name__ == "__main__":
    unittest.main()

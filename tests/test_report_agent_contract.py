"""Contract tests for the report pipeline, using fixed evidence and mocked LLM output."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
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
The evidence provides the scaled dot-product attention formulation [1]. Details on masking variants and runtime remain unavailable [1].

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

    def test_explicit_synthesis_caveat_prevents_false_fully_covered_status(self):
        context = fixture_context()
        question = "Where is the official TensorFlow attention API documented?"
        context["planner_questions"] = [question]
        context["per_question_synthesis"][0].update({
            "question": question,
            "coverage": "covered",
            "synthesis": "PyTorch documents MultiheadAttention [1]. Exact missing details: The TensorFlow API location is not provided.",
        })
        context["evidence_packs"][0]["coverage"] = "covered"
        context["evidence_packs"][0]["question"] = question
        contracts = report_agent.build_question_evidence_contracts(
            [question],
            report_agent.per_question_synthesis_by_question(context["per_question_synthesis"]),
            {report_agent.normalize_heading(question): context["evidence_packs"][0]},
            context["sources"],
        )

        self.assertEqual(contracts[report_agent.normalize_heading(question)]["coverage"], "partial")

    def test_unrelated_caveat_does_not_downgrade_question_pack_coverage(self):
        context = fixture_context()
        context["per_question_synthesis"][0]["coverage"] = "partial"
        context["evidence_packs"][0]["coverage"] = "covered"
        contracts = report_agent.build_question_evidence_contracts(
            [QUESTION],
            report_agent.per_question_synthesis_by_question(context["per_question_synthesis"]),
            {report_agent.normalize_heading(QUESTION): context["evidence_packs"][0]},
            context["sources"],
        )

        self.assertEqual(contracts[report_agent.normalize_heading(QUESTION)]["coverage"], "covered")

    def test_revision_prompt_is_compact_and_keeps_every_question_contract(self):
        context = fixture_context()
        contracts = report_agent.build_question_evidence_contracts(
            [QUESTION],
            report_agent.per_question_synthesis_by_question(context["per_question_synthesis"]),
            {report_agent.normalize_heading(QUESTION): context["evidence_packs"][0]},
            context["sources"],
        )
        prompt = report_agent.build_report_revision_prompt(
            {"issues": ["missing benchmark evidence"], "schema_issues": []},
            GOOD_REPORT * 8,
            [QUESTION],
            contracts,
        )

        self.assertLess(len(prompt), 9000)
        self.assertIn(QUESTION, prompt)
        self.assertIn("runtime measurements", prompt)
        self.assertIn("Draft to repair", prompt)

    def test_schema_validator_rejects_topic_number_assigned_to_wrong_question(self):
        questions = [
            "What is the definition of attention?",
            "What are the benchmark results for attention?",
        ]
        report = GOOD_REPORT.replace(
            "### 3.1. Mathematical Formulation Of Scaled Dot-product Attention Including Equation",
            "### 3.1. Benchmark Results",
        ).replace(
            "## 4. Cross-cutting Analysis and Synthesis",
            "### 3.2. Definition Of Attention\nAttention assigns weights to input information [1].\n\n## 4. Cross-cutting Analysis and Synthesis",
        )

        issues = report_agent.report_schema_issues(report, questions)

        self.assertTrue(any("3.1" in issue and "definition" in issue.lower() for issue in issues))
        self.assertTrue(any("3.2" in issue and "benchmark" in issue.lower() for issue in issues))

    def test_schema_accepts_specific_benchmark_heading(self):
        question = "How do attention mechanisms perform on standard benchmarks such as machine translation (WMT) and GLUE?"

        self.assertTrue(report_agent.topic_heading_matches_question("Benchmark Performance (WMT & GLUE)", question))

    def test_gap_caveat_does_not_bypass_required_supported_methods(self):
        question = "What are the known limitations and recent efficient variants of attention mechanisms?"
        section = (
            "### 3.1. Limitations and Efficient Variants\n"
            "Sparse attention reduces token-pair computations [2].\n"
            "The evidence does not identify other efficient approaches."
        )

        issues = report_agent.required_evidence_issues(
            f"## 3. Topic Sections\n{section}", [question]
        )

        self.assertTrue(any("efficient attention approaches" in issue for issue in issues))

    def test_supported_methods_and_specific_gap_satisfy_efficient_variant_contract(self):
        question = "What are the known limitations and recent efficient variants of attention mechanisms?"
        section = (
            "Sparse attention reduces token-pair computations [2].\n"
            "Kernel-based linear approximations reduce quadratic computation [2].\n"
            "The evidence does not provide comparative benchmark results."
        )

        self.assertTrue(report_agent.section_satisfies_required_evidence(question, section))

    def test_cited_table_finding_prevents_false_total_gap(self):
        question = "What are the main types of attention mechanisms and their equations?"
        report = (
            "## 3. Topic Sections\n### 3.1. Main Types of Attention\n"
            "| Type | Supported description |\n| --- | --- |\n"
            "| Additive | Uses a feed-forward compatibility function [1] |\n"
            "Self-attention's exact equation is not provided."
        )
        synthesis = [{"question": question, "synthesis": "Additive attention uses a feed-forward scoring function [1] and multiplicative attention uses dot products [2]."}]

        self.assertEqual(
            report_agent.report_evidence_gap_contradictions(report, [question], synthesis), []
        )

    def test_context_labels_keep_application_bullets_on_topic(self):
        question = "What are the primary applications of attention mechanisms in NLP and computer vision?"
        report = (
            "## 3. Topic Sections\n### 3.1. Primary Applications\n"
            "**Natural-Language Processing**\n- Language modeling with autoregressive Transformers [2]\n"
            "**Computer Vision**\n- Vision Transformers for image classification [2]"
        )

        self.assertEqual(report_agent.report_topic_relevance_issues(report, [question]), [])

    def test_framing_repair_rebuilds_uncited_sections_from_cited_topic_content(self):
        report = GOOD_REPORT.replace(
            "This report examines the formula and its documented limitations [1].",
            "Attention is widely used for many tasks without citation.",
        ).replace(
            "The evidence provides the scaled dot-product attention formulation [1]. Details on masking variants and runtime remain unavailable [1].",
            "This report considers attention mechanisms and their applications without citation.",
        )

        repaired, repairs = report_agent.repair_uncited_frame_sections(report, fixture_context()["sources"])

        self.assertTrue(repairs)
        self.assertEqual(report_agent.uncited_factual_frame_sections(repaired), [])
        self.assertNotIn("without citation", repaired)

    def test_explanatory_where_clause_after_equation_is_not_an_artifact(self):
        clause = r"where (Q) (queries), (K) (keys), and (V) (values) are matrices [1]."

        self.assertTrue(report_agent.is_equation_definition_clause(clause))
        self.assertNotIn(clause, report_agent.report_artifact_lines(clause))

    def test_bahdanau_equation_present_in_synthesis_must_be_preserved(self):
        question = "What is the mathematical formulation of Bahdanau additive attention?"
        synthesis = [{
            "question": question,
            "synthesis": r"The score is \(a(s,h)=v_a^T\\tanh(W_a s+U_a h)\) [1].",
        }]
        incomplete = (
            "## 3. Topic Sections\n### 3.1. Bahdanau Additive Attention\n"
            "The score uses a learned additive function [1]."
        )
        complete = incomplete + r"\n\nCore equation: \(a(s,h)=v_a^T\\tanh(W_a s+U_a h)\) [1]."

        self.assertEqual(len(report_agent.supported_equation_omissions(incomplete, [question], synthesis)), 1)
        self.assertEqual(report_agent.supported_equation_omissions(complete, [question], synthesis), [])

    def test_supported_display_equations_survive_topic_extraction(self):
        question = "What equations define additive attention, including the score and context vector?"
        note = {
            "source_indexes": [1],
            "synthesis": (
                r"The score is defined as \(e_{ij}=v_a^T\tanh(W_a s_{i-1}+U_a h_j)\) [1]. "
                r"Weights are normalized as \(\alpha_{ij}=\operatorname{softmax}(e_{ij})\) [1]. "
                r"The context is \(c_i=\sum_j\alpha_{ij}h_j\) [1]."
            ),
        }

        parts = report_agent.topic_body_from_synthesis(question, note)
        body = "\n".join(parts)

        self.assertGreaterEqual(body.count("Core equation"), 3)
        self.assertIn(r"e_{ij}=v_a^T", body)
        self.assertIn(r"\alpha_{ij}=", body)
        self.assertIn(r"c_i=", body)
        self.assertEqual(report_agent.supported_equation_omissions(
            f"## 3. Topic Sections\n### 3.1. Additive Attention Equations\n{body}", [question], [note]
        ), [])

    def test_supported_prose_is_not_replaced_by_gap_when_detail_check_fails(self):
        question = "What are the benchmark results for attention on WMT and GLUE?"
        pack = {"coverage": "partial", "chunks": []}
        note = {
            "source_indexes": [1],
            "synthesis": "The Transformer reported 28.4 BLEU on WMT 2014 English-to-German [1]. Exact missing details: GLUE results are not reported.",
        }

        section, _ = report_agent.build_topic_section(
            1, question, pack, note, [{"index": 1, "url": "https://example.org/paper"}]
        )

        self.assertIn("28.4 BLEU", section)
        self.assertIn("GLUE", section)
        self.assertNotIn("did not provide enough benchmark", section)

    def test_gap_only_draft_does_not_satisfy_equation_or_benchmark_coverage(self):
        for question in (
            "What is the mathematical equation for attention?",
            "What benchmark results demonstrate its performance?",
        ):
            self.assertFalse(report_agent.specialized_question_covered(
                question, "Evidence is incomplete; the requested detail is missing."
            ))

    def test_standalone_synthesis_citations_attach_to_supported_claims(self):
        question = "What are the limitations and recent solutions for attention?"
        note = {
            "source_indexes": [1],
            "synthesis": "**Supported statements**\n- Self-attention has quadratic time and memory complexity.\n  *Citation:* [1]\n- Efficient X-former models address scalability.\n  *Citation:* [1]\n\n**Missing details**\n- Specific architecture comparisons are not available.",
        }

        body = "\n".join(report_agent.topic_body_from_synthesis(question, note))

        self.assertIn("quadratic time and memory complexity. [1]", body)
        self.assertIn("X-former models address scalability [1].", body)
        self.assertTrue(report_agent.section_satisfies_required_evidence(question, body))

    def test_token_truncated_llm_draft_is_retained_for_revision(self):
        choice = type("Choice", (), {
            "message": type("Message", (), {"content": GOOD_REPORT[:250]})(),
            "finish_reason": "length",
        })()
        response = type("Response", (), {"choices": [choice], "model": "mock"})()
        fake_sdk = SimpleNamespace(Groq=lambda: object())
        with patch.dict("os.environ", {"GROQ_API_KEY": "test"}), \
             patch.dict(sys.modules, {"groq": fake_sdk}), \
             patch.object(report_agent, "create_chat_completion_with_retries", return_value=response):
            draft, mode, error = report_agent.generate_report_with_llm("mock", "prompt")

        self.assertEqual(draft, GOOD_REPORT[:250])
        self.assertEqual(mode, "llm_truncated_partial")
        self.assertIn("token limit", error)

    def test_truncated_report_triggers_revision_instead_of_deterministic_fallback(self):
        with patch.object(
            report_agent, "generate_report_with_llm",
            side_effect=[
                (GOOD_REPORT[:250], "llm_truncated_partial", "LLM response reached its token limit"),
                (GOOD_REPORT, "mocked_llm", ""),
            ],
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertTrue(payload["diagnostics"]["report_llm_revision"]["attempted"])
        self.assertTrue(payload["diagnostics"]["report_llm_revision"]["accepted"])
        self.assertFalse(payload["diagnostics"]["report_deterministic_fallback_used"])
        self.assertEqual(payload["diagnostics"]["report_generation_mode"], "llm_revised_from_question_evidence")

    def test_placeholder_citation_is_removed_during_cleanup(self):
        cleaned, repairs = report_agent.cleanup_report(
            "## 3.1. Attention Types\nSelf-attention's equation is not supplied [—].",
            fixture_context()["sources"],
        )

        self.assertNotIn("[—]", cleaned)
        self.assertIn("removed placeholder citation marker", repairs)

    def test_saved_quality_regression_fixture_flags_known_report_defects(self):
        fixture_path = Path(__file__).parent / "fixtures" / "report_quality_regression.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        validation = report_agent.validate_report(
            fixture["draft"], fixture["sources"], fixture["questions"],
            fixture["evidence_packs"], fixture["per_question_synthesis"],
        )

        self.assertEqual(validation["schema_issues"], [])
        self.assertEqual(validation["false_gap_questions"], [])
        self.assertTrue(any("Bahdanau scoring equation" in issue for issue in validation["issues"]))
        self.assertTrue(any("efficient attention approaches" in issue for issue in validation["issues"]))

    def test_revision_gate_rejects_changes_to_unaffected_topic_sections(self):
        second_question = "What are the primary applications of attention mechanisms?"
        original = GOOD_REPORT.replace(
            "## 4. Cross-cutting Analysis and Synthesis",
            "### 3.2. Primary Applications\nAttention is applied to image classification [2].\n\n## 4. Cross-cutting Analysis and Synthesis",
        )
        unrelated_change = original.replace(
            "Attention is applied to image classification [2].",
            "Applications include image classification and detection [2].",
        )
        targeted_change = original.replace(
            "Scaled dot-product attention normalizes query-key scores",
            "The scaled dot-product equation normalizes query-key scores",
        )
        validation = {"issues": [f"missing formulation: {QUESTION}"], "schema_issues": []}

        self.assertFalse(report_agent.revision_preserves_unaffected_topics(
            original, unrelated_change, [QUESTION, second_question], validation
        ))
        self.assertTrue(report_agent.revision_preserves_unaffected_topics(
            original, targeted_change, [QUESTION, second_question], validation
        ))

    def test_generate_rejects_revision_that_improves_score_but_rewrites_unaffected_topic(self):
        bad = GOOD_REPORT.replace(
            "This report examines the formula and its documented limitations [1].",
            "This report examines the supplied material without a citation.",
        )
        revised = GOOD_REPORT.replace(
            "Scaled dot-product attention normalizes query-key scores",
            "The equation rescales query-key scores",
        )
        with patch.object(
            report_agent, "generate_report_with_llm",
            side_effect=[(bad, "mocked_llm", ""), (revised, "mocked_llm", "")],
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        revision = payload["diagnostics"]["report_llm_revision"]
        self.assertFalse(revision["accepted"])
        self.assertIn("unrelated", revision["rejection_reason"])
        self.assertIn("Scaled dot-product attention normalizes", payload["report"])

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
        self.assertTrue(payload["diagnostics"]["report_llm_used"])
        self.assertFalse(payload["diagnostics"]["report_deterministic_fallback_used"])

    def test_invalid_llm_draft_is_revised_by_llm_not_replaced_with_deterministic_prose(self):
        bad_report = GOOD_REPORT.replace(
            "This report examines the formula and its documented limitations [1].",
            "Attention is widely used for sequence tasks without citation.",
        )
        with patch.object(
            report_agent, "generate_report_with_llm",
            side_effect=[(bad_report, "mocked_llm", ""), (GOOD_REPORT, "mocked_llm", "")],
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertTrue(payload["diagnostics"]["report_llm_revision"]["attempted"])
        self.assertTrue(payload["diagnostics"]["report_llm_revision"]["accepted"])
        self.assertEqual(payload["diagnostics"]["report_issues"], [])
        self.assertEqual(payload["diagnostics"]["report_generation_mode"], "llm_revised_from_question_evidence")

    def test_failed_llm_revision_keeps_draft_and_marks_needs_review(self):
        bad_report = GOOD_REPORT.replace(
            "This report examines the formula and its documented limitations [1].",
            "Attention is widely used for sequence tasks without citation.",
        )
        with patch.object(
            report_agent, "generate_report_with_llm",
            side_effect=[(bad_report, "mocked_llm", ""), (None, "deterministic_fallback_llm_error", "revision unavailable")],
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertNotIn("Attention is widely used for sequence tasks", payload["report"])
        self.assertTrue(any("uncited" in item for item in payload["diagnostics"]["report_deterministic_repairs"]))
        self.assertTrue(payload["diagnostics"]["report_llm_used"])
        self.assertFalse(payload["diagnostics"]["report_deterministic_fallback_used"])
        self.assertEqual(payload["diagnostics"]["report_finalization_status"], "clean_with_evidence_gaps")
        self.assertEqual(payload["diagnostics"]["report_llm_revision"]["error"], "revision unavailable")

    def test_all_gap_topic_fails_when_question_synthesis_has_supported_claims(self):
        gap_report = (
            "## 3. Topic Sections\n"
            "### 3.1. Mathematical Formulation Of Scaled Dot-product Attention Including Equation\n"
            "The retrieved evidence is incomplete for this sub-question."
        )
        validation = report_agent.validate_report(
            gap_report,
            fixture_context()["sources"],
            [QUESTION],
            fixture_context()["evidence_packs"],
            fixture_context()["per_question_synthesis"],
        )

        self.assertEqual(validation["false_gap_questions"], [QUESTION])

    def test_targeted_topic_repair_replaces_gap_only_section_when_question_has_evidence(self):
        context = fixture_context()
        gap_report = (
            "## 3. Topic Sections\n"
            "### 3.1. Mathematical Formulation Of Scaled Dot-product Attention Including Equation\n"
            "The retrieved evidence is incomplete for this sub-question."
        )
        notes = report_agent.per_question_synthesis_by_question(context["per_question_synthesis"])
        packs = {report_agent.normalize_heading(item["question"]): item for item in context["evidence_packs"]}

        repaired, diagnostics = report_agent.repair_report_topic_sections(
            gap_report, [QUESTION], packs, notes, context["sources"]
        )

        self.assertTrue(diagnostics)
        self.assertIn("Core equation", repaired)
        self.assertIn("softmax", repaired)
        self.assertNotIn("evidence is incomplete", repaired.lower())

    def test_all_gap_topic_fails_when_only_question_pack_has_clean_evidence(self):
        gap_report = (
            "## 3. Topic Sections\n"
            "### 3.1. Mathematical Formulation Of Scaled Dot-product Attention Including Equation\n"
            "The retrieved evidence is incomplete for this sub-question."
        )
        pack = {
            "question": QUESTION,
            "chunks": [{
                "source_index": 2,
                "content": "Scaled dot-product attention computes query-key scores, applies softmax, and weights values.",
            }],
        }

        contradictions = report_agent.report_evidence_gap_contradictions(gap_report, [QUESTION], [], [pack])

        self.assertEqual(contradictions, [QUESTION])

    def test_deterministic_fallback_does_not_copy_unrelated_chunk_fragments(self):
        question = "What is the definition of the attention mechanism in machine learning?"
        section, _ = report_agent.build_topic_section(
            1,
            question,
            {
                "coverage": "covered",
                "chunks": [{
                    "source_index": 1,
                    "content": "Semantic plausibility (animals get tired, streets do not) 3 [1]. g, attention mechanisms have enabled breakthroughs in protein structure prediction [1].",
                }],
            },
            {},
            [{"index": 1, "url": "https://example.org/source"}],
        )

        self.assertNotIn("Semantic plausibility", section)
        self.assertNotIn("g, attention", section)
        self.assertIn("evidence", section.lower())

    def test_deterministic_fallback_drops_keyword_metadata_chunks(self):
        sentence = report_agent.chunk_to_evidence_sentence(
            "What is the definition of the attention mechanism in neural networks?",
            {"source_index": 1, "content": "Keywords: Deep Learning, Natural Language Processing, Transformer Models, Attention Models, Neural Networks."},
        )

        self.assertEqual(sentence, "")

    def test_llm_failure_returns_deterministic_report_instead_of_raising(self):
        with patch.object(
            report_agent, "generate_report_with_llm",
            return_value=(None, "deterministic_fallback_llm_error", "mock model unavailable"),
        ):
            payload = report_agent.ReportAgent(model="mocked-model").generate(fixture_context())

        self.assertTrue(payload["report"])
        self.assertIn("## 6. Conclusion", payload["report"])
        self.assertEqual(payload["diagnostics"]["report_generation_error"], "mock model unavailable")
        self.assertFalse(payload["diagnostics"]["report_llm_used"])
        self.assertTrue(payload["diagnostics"]["report_deterministic_fallback_used"])


if __name__ == "__main__":
    unittest.main()

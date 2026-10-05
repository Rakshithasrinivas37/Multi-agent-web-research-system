"""Regression checks for evidence lost between retrieval and prompt construction."""

import unittest
from unittest.mock import patch

from src.rag import generation as g
from src.rag.retrieval import RetrievalResult


def result(identifier, text, **metadata):
    return RetrievalResult(id=identifier, document=text,
                           metadata={"url": "https://example.org/paper", **metadata},
                           score=1.0, semantic_score=0.8, bm25_score=0.2)


class EvidenceSelectionTests(unittest.TestCase):
    def test_boilerplate_does_not_win_by_listing_all_facets(self):
        overview = result("overview", "We begin by establishing additive, multiplicative, scaled dot-product and multi-head attention. The Transformer architecture is presented with complete mathematical derivations of all components.")
        answer = result("answer", "Additive attention computes compatibility using a feed-forward network with a single hidden layer. Multiplicative attention uses a dot product between query and key vectors.")
        trace = []
        _, selected = g.build_question_context_evidence(
            [overview, answer], [], question="Compare attention (additive, multiplicative, multi-head).",
            max_chunks=1, selection_trace=trace)
        self.assertEqual([c["id"] for c in selected], ["answer"])
        self.assertIn({"id": "overview", "outcome": "boilerplate"}, trace)

    def test_fallback_uses_exact_prompt_evidence_and_records_gaps(self):
        equation = "MultiHead(Q,K,V) = Concat(head1, head2)WO."
        chunk = result("equation", "Multi-head attention performs attention in parallel with learned projections and combines the results using an output projection. " + equation)
        _, sources = g.build_generation_context([chunk])
        with patch.object(g, "create_chat_completion_with_retries", side_effect=RuntimeError("offline")), \
             patch("builtins.print"):
            note = g.synthesize_per_question_notes(
                object(), "test", "Attention", "Summarize", ["Compare attention (multi-head, Performer)."],
                [chunk], sources)[0]
        self.assertIn(equation, note["synthesis"])
        self.assertTrue(note["fallback_used"])
        self.assertTrue(any("performer" in f.lower() for f in note["selection_trace"]["missing_facets"]))
        self.assertIn(note["selected_chunks"][0]["content"], note["synthesis"])

    def test_equation_after_old_excerpt_limit_survives(self):
        equation = "MultiHead(Q,K,V) = Concat(head1, head2)WO where headi = Attention(QWi, KWi, VWi)."
        body = "The projected values are combined in parallel to represent multiple relationships. " * 10 + equation
        chunk = result("equation", body)
        context, selected = g.build_question_context_evidence(
            [chunk], [], question="What is the multi-head attention equation?", max_chars=2200)
        self.assertIn(equation, context)
        self.assertEqual(len(selected), 1)

    def test_duplicates_do_not_consume_slots_and_tail_refills(self):
        body = "Translation benchmarks report a score of 28.4 BLEU on the English German dataset with the Transformer model trained under the reported experimental conditions."
        chunks = [result(str(i), body) for i in range(5)]
        chunks.append(result("vision", "Vision Transformer achieved 88 percent accuracy on the ImageNet benchmark in the reported evaluation using the specified training dataset and configuration."))
        trace = []
        context, selected = g.build_question_context_evidence(
            chunks, [], question="What benchmark results are reported?", max_chunks=2,
            selection_trace=trace)
        self.assertEqual([c["id"] for c in selected], ["0", "vision"])
        self.assertEqual(sum(d["outcome"] == "duplicate_excerpt" for d in trace), 4)
        self.assertIn("ImageNet", context)

    def test_metadata_does_not_make_unrelated_body_relevant(self):
        wrong = result("wrong", "The orchard grows apples and pears throughout summer and supplies fresh fruit to nearby villages. Its trees are watered regularly during dry weather.",
                       query_contexts="attention mechanism", title="Attention mechanism")
        self.assertEqual(g.rank_results_for_question("What is attention?", [wrong]), [])

    def test_comparison_beats_approximation_errors(self):
        question = "What types of attention mechanisms are additive and multiplicative and how do they differ?"
        wrong = result("wrong", "Additive approximation errors and multiplicative approximation errors give bounds for exact attention computation.")
        right = result("right", "Additive attention computes compatibility with a feed-forward network while multiplicative attention uses a dot product between queries and keys.")
        self.assertEqual(g.rank_results_for_question(question, [wrong, right])[0].id, "right")

    def test_oversized_unit_is_not_clipped_and_next_chunk_can_fit(self):
        oversized = result("large", "equation = " + "symbol " * 400)
        small = result("small", "Attention combines value vectors using normalized weights derived from the compatibility between queries and keys to produce a contextual representation.")
        context, selected = g.build_question_context_evidence(
            [oversized, small], [], question="What is attention?", max_chars=400)
        self.assertLessEqual(len(context), 400)
        self.assertEqual([c["id"] for c in selected], ["small"])


if __name__ == "__main__":
    unittest.main()

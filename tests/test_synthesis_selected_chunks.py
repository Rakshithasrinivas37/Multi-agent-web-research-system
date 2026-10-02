"""Selected synthesis evidence must survive the shared-memory handoff."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.agents.synthesis_agent import SynthesisAgent
from src.rag import generation
from src.rag.retrieval import RetrievalResult


class SynthesisSelectedChunksTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            RetrievalResult(
                id=f"chunk-{i}",
                document=f"Evidence number {i} explains how attention weights combine query and key representations. " * 3,
                metadata={"title": f"Paper {i}", "url": f"https://example.org/paper-{i}"},
                score=1.0 / i, semantic_score=0.8, bm25_score=0.2,
            )
            for i in range(1, 7)
        ]
        _, self.sources = generation.build_generation_context(self.results)
        self.response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Attention combines representations [1].")
        )])

    def synthesize(self, questions):
        return generation.synthesize_per_question_notes(
            object(), "mock-model", "Attention", "Summarize evidence",
            questions, self.results, self.sources,
        )

    def test_selected_prompt_chunks_persist_for_each_question(self):
        with patch.object(generation, "rank_results_for_question", side_effect=[self.results, self.results[::-1]]), \
             patch.object(generation, "create_chat_completion_with_retries", return_value=self.response) as complete:
            notes = self.synthesize(["What is attention?", "How do attention weights work?"])

        for note, call, expected in zip(notes, complete.call_args_list, [self.results[:4], self.results[::-1][:4]]):
            chunks = note["selected_chunks"]
            self.assertEqual(note["selected_chunk_count"], 4)
            self.assertEqual([chunk["id"] for chunk in chunks], [result.id for result in expected])
            prompt = call.kwargs["messages"][1]["content"]
            for chunk, result in zip(chunks, expected):
                self.assertIn(chunk["content"], prompt)
                self.assertIn(f"[{chunk['source_index']}] {chunk['title']}", prompt)
                self.assertEqual(chunk["url"], result.metadata["url"])
                self.assertEqual(chunk["score"], result.score)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared_memory.json"
            SynthesisAgent().write_to_memory({"per_question_synthesis": notes}, str(path))
            saved = json.loads(path.read_text())["synthesis"]["report_context"]["per_question_synthesis"]
        self.assertEqual(saved, notes)

    def test_budget_excluded_chunks_are_not_saved_as_prompt_evidence(self):
        context, chunks = generation.build_question_context_evidence(self.results[:4], self.sources, max_chars=400)
        self.assertEqual(len(chunks), 1)
        self.assertIn(chunks[0]["content"], context)
        self.assertNotIn("Paper 2", context)
        self.assertEqual(context, generation.build_question_context_text(self.results[:4], self.sources, max_chars=400))

    def test_empty_selection_records_empty_list_without_llm_call(self):
        with patch.object(generation, "rank_results_for_question", return_value=[]), \
             patch.object(generation, "create_chat_completion_with_retries") as complete:
            note = self.synthesize(["What is missing?"])[0]
        complete.assert_not_called()
        self.assertEqual(note["selected_chunks"], [])
        self.assertEqual(note["selected_chunk_count"], 0)
        self.assertEqual(note["context_chars"], 0)

    def test_failure_keeps_evidence_sent_in_attempted_prompt(self):
        with patch.object(generation, "rank_results_for_question", return_value=self.results), \
             patch.object(generation, "create_chat_completion_with_retries", side_effect=RuntimeError("service unavailable")):
            note = self.synthesize(["What is attention?"])[0]
        self.assertTrue(note["fallback_used"])
        self.assertEqual(note["selected_chunk_count"], 4)
        self.assertEqual(note["error"], "service unavailable")


if __name__ == "__main__":
    unittest.main()

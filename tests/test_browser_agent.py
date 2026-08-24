import unittest

from src.agents.browser_agent import (
    clean_source_content,
    search_result_score,
    source_is_useful,
    source_quality_label,
    source_quality_note,
)


class BrowserAgentQualityTests(unittest.TestCase):
    def test_direct_arxiv_source_is_useful_even_with_generic_target_name(self):
        task = {
            "url": "https://arxiv.org/abs/1409.0473",
            "query_context": "What are the key contributions of the research paper?",
            "extraction_goal": "summary of method, evidence, and experimental impact",
            "target_name": "Research Topic",
            "expected_signals": ["method", "evidence", "results"],
        }
        payload = {
            "url": "https://arxiv.org/pdf/1409.0473",
            "content_length": 50025,
            "content_preview": "Research paper with method evidence and experimental results.",
            "full_content": "Research paper with method evidence and experimental results. " * 20,
        }

        self.assertTrue(source_is_useful(payload, task))
        self.assertEqual(source_quality_note(payload, task), "trusted direct authoritative source")

    def test_direct_authoritative_source_still_rejects_short_content(self):
        task = {
            "url": "https://arxiv.org/abs/1409.0473",
            "query_context": "Research paper",
            "target_name": "Research Topic",
        }
        payload = {
            "url": "https://arxiv.org/pdf/1409.0473",
            "content_length": 100,
            "content_preview": "",
            "full_content": "",
        }

        self.assertFalse(source_is_useful(payload, task))
        self.assertEqual(source_quality_note(payload, task), "source content is empty, blocked, or too short")

    def test_clean_source_content_removes_common_boilerplate(self):
        content = """
        Accept cookies
        Sign up
        This report explains the target method, benchmark evidence, and implementation details.
        Privacy policy
        This section includes source-specific findings that should be indexed.
        """

        cleaned = clean_source_content(content)

        self.assertIn("benchmark evidence", cleaned)
        self.assertIn("source-specific findings", cleaned)
        self.assertNotIn("Accept cookies", cleaned)
        self.assertNotIn("Privacy policy", cleaned)

    def test_search_result_score_prefers_authoritative_sources(self):
        task = {
            "query_context": "Find API documentation and implementation examples.",
            "extraction_goal": "Extract API usage.",
            "target_name": "General Research",
            "source_type": "search",
        }

        docs_score = search_result_score(task, "https://docs.example.org/api/reference")
        forum_score = search_result_score(task, "https://forum.example.org/tag/api")

        self.assertGreater(docs_score, forum_score)

    def test_source_quality_label_preserves_useful_authority_signal(self):
        task = {
            "url": "https://example.edu/research/page",
            "query_context": "Find benchmark evidence.",
            "extraction_goal": "Extract benchmark evidence and examples.",
            "target_name": "General Research",
            "expected_signals": ["benchmark evidence"],
        }
        payload = {
            "url": "https://example.edu/research/page",
            "content_length": 1500,
            "content_noise_score": 0.0,
            "content_preview": "Benchmark evidence and examples from a university source.",
            "full_content": "Benchmark evidence and examples from a university source. " * 30,
            "extracted_facts": ["Benchmark evidence and examples from a university source."],
            "source_authority": "authoritative",
        }

        self.assertEqual(source_quality_label(payload, task), "useful_authoritative")


if __name__ == "__main__":
    unittest.main()

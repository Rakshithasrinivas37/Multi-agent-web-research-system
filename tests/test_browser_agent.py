import unittest

from src.agents.browser_agent import source_is_useful, source_quality_note


class BrowserAgentQualityTests(unittest.TestCase):
    def test_direct_arxiv_source_is_useful_even_with_generic_target_name(self):
        task = {
            "url": "https://arxiv.org/abs/1409.0473",
            "query_context": "What are the key contributions of the Bahdanau attention paper?",
            "extraction_goal": "summary of method, additive attention, experimental impact",
            "target_name": "Attention Mechanism",
            "expected_signals": ["additive attention", "alignment", "BLEU"],
        }
        payload = {
            "url": "https://arxiv.org/pdf/1409.0473",
            "content_length": 50025,
            "content_preview": "Neural Machine Translation by Jointly Learning to Align and Translate.",
            "full_content": "Neural Machine Translation by Jointly Learning to Align and Translate. "
            "Bahdanau Cho Bengio encoder decoder alignment translation." * 20,
        }

        self.assertTrue(source_is_useful(payload, task))
        self.assertEqual(source_quality_note(payload, task), "trusted direct authoritative source")

    def test_direct_authoritative_source_still_rejects_short_content(self):
        task = {
            "url": "https://arxiv.org/abs/1409.0473",
            "query_context": "Bahdanau attention paper",
            "target_name": "Attention Mechanism",
        }
        payload = {
            "url": "https://arxiv.org/pdf/1409.0473",
            "content_length": 100,
            "content_preview": "",
            "full_content": "",
        }

        self.assertFalse(source_is_useful(payload, task))
        self.assertEqual(source_quality_note(payload, task), "source content is empty, blocked, or too short")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from src.agents.synthesis_agent import SynthesisAgent


class SynthesisAgentTests(unittest.TestCase):
    def test_synthesize_passes_browser_results_to_generation(self):
        browser_results = [{"sources": [{"url": "https://example.com", "full_content": "Formula: y = mx + b."}]}]
        plan = {"objective": "Test objective"}

        with patch("src.agents.synthesis_agent.synthesize_report_from_research_plan") as synthesize:
            synthesize.return_value = {"objective": "Test objective", "synthesis": "notes", "sources": []}

            payload = SynthesisAgent().synthesize(plan, browser_results=browser_results)

        self.assertEqual(payload["synthesis"], "notes")
        self.assertEqual(synthesize.call_args.kwargs["browser_results"], browser_results)

    def test_write_to_memory_strips_thinking_blocks(self):
        with patch("src.agents.synthesis_agent.SharedMemory") as memory_cls:
            SynthesisAgent().write_to_memory({"synthesis": "Useful.\n<think>hidden</think>\nFinal."})

        written = memory_cls.return_value.write_agent_output.call_args.args[1]["report_context"]["synthesis"]
        self.assertEqual(written, "Useful.\n\nFinal.")


if __name__ == "__main__":
    unittest.main()

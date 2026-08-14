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


if __name__ == "__main__":
    unittest.main()

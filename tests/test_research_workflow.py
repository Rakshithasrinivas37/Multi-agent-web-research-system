import unittest
from unittest.mock import patch

from langgraph.graph import END

from src.graph.research_workflow import DEFAULT_REPORT_GAP_SYNTHESIS_MODEL, next_node_or_end, report_gap_synthesis_model


class ResearchWorkflowRoutingTests(unittest.TestCase):
    def test_next_node_or_end_routes_to_next_node_without_errors(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": []}), "browser")

    def test_next_node_or_end_stops_when_errors_exist(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": ["planner failed"]}), END)

    @patch.dict("os.environ", {}, clear=True)
    def test_report_gap_synthesis_model_defaults_to_qwen(self):
        self.assertEqual(report_gap_synthesis_model(), DEFAULT_REPORT_GAP_SYNTHESIS_MODEL)

    @patch.dict("os.environ", {"REPORT_GAP_SYNTHESIS_MODEL": "custom/model"}, clear=True)
    def test_report_gap_synthesis_model_allows_env_override(self):
        self.assertEqual(report_gap_synthesis_model(), "custom/model")


if __name__ == "__main__":
    unittest.main()

import unittest

from langgraph.graph import END

from src.graph.research_workflow import next_node_or_end


class ResearchWorkflowRoutingTests(unittest.TestCase):
    def test_next_node_or_end_routes_to_next_node_without_errors(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": []}), "browser")

    def test_next_node_or_end_stops_when_errors_exist(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": ["planner failed"]}), END)


if __name__ == "__main__":
    unittest.main()

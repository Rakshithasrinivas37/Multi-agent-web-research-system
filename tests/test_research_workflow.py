import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from langgraph.graph import END

from src.graph.research_workflow import (
    DEFAULT_REPORT_GAP_SYNTHESIS_MODEL,
    format_exception_details,
    next_node_or_end,
    print_retry_response_error,
    report_gap_synthesis_model,
    validate_rag_index,
)


class ResearchWorkflowRoutingTests(unittest.TestCase):
    def test_next_node_or_end_routes_to_next_node_without_errors(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": []}), "browser")

    def test_next_node_or_end_stops_when_errors_exist(self):
        route = next_node_or_end("browser")

        self.assertEqual(route({"errors": ["planner failed"]}), END)

    def test_format_exception_details_includes_traceback(self):
        try:
            raise RuntimeError("full error")
        except RuntimeError as error:
            details = format_exception_details(error)

        self.assertIn("Traceback", details)
        self.assertIn("RuntimeError: full error", details)

    def test_print_retry_response_error_preserves_multiline_error(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_retry_response_error("synthesis", 1, ["line one\nline two"])

        output = buffer.getvalue()
        self.assertIn("[synthesis] retrying after response error", output)
        self.assertIn("line one\nline two", output)

    @patch.dict("os.environ", {}, clear=True)
    def test_report_gap_synthesis_model_defaults_to_qwen(self):
        self.assertEqual(report_gap_synthesis_model(), DEFAULT_REPORT_GAP_SYNTHESIS_MODEL)

    @patch.dict("os.environ", {"REPORT_GAP_SYNTHESIS_MODEL": "custom/model"}, clear=True)
    def test_report_gap_synthesis_model_allows_env_override(self):
        self.assertEqual(report_gap_synthesis_model(), "custom/model")

    def test_validate_rag_index_allows_reused_existing_chunks(self):
        index = {"status": "success", "indexed_chunks": 0, "stored_chunks": 12, "skipped_sources": 3}

        self.assertEqual(validate_rag_index(index), [])

    def test_validate_rag_index_rejects_empty_new_and_stored_chunks(self):
        index = {"status": "success", "indexed_chunks": 0, "stored_chunks": 0}

        self.assertEqual(validate_rag_index(index), ["rag_index indexed zero chunks"])


if __name__ == "__main__":
    unittest.main()

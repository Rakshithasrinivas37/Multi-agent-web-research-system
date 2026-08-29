import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from langgraph.graph import END

from src.graph.research_workflow import (
    DEFAULT_REPORT_GAP_SYNTHESIS_MODEL,
    format_exception_details,
    mark_report_gap_refresh_empty,
    next_node_or_end,
    print_retry_response_error,
    report_gap_refresh_has_context,
    report_gap_synthesis_model,
    research_plan_for_report_gaps,
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

    def test_research_plan_for_report_gaps_keeps_matching_specs(self):
        question = "What is the scaled dot-product attention equation?"
        plan = {
            "objective": "Attention mechanism",
            "sub_questions": [question, "What is self-attention?"],
            "sub_question_specs": [
                {"question_id": "q001", "question": question, "required_evidence": ["equation"]},
                {"question_id": "q002", "question": "What is self-attention?", "required_evidence": ["definition"]},
            ],
        }

        gap_plan = research_plan_for_report_gaps(plan, [question], ["scaled dot product attention equation"])

        self.assertEqual(gap_plan["sub_questions"], [question])
        self.assertEqual(gap_plan["sub_question_specs"], [plan["sub_question_specs"][0]])
        self.assertEqual(gap_plan["tasks"][0]["query_context"], question)
        self.assertEqual(gap_plan["tasks"][0]["url"], "scaled dot product attention equation")

    def test_empty_report_gap_refresh_is_marked_without_overwriting_context(self):
        original = {"synthesis": "existing synthesis", "diagnostics": {"existing": True}}
        refreshed = {
            "retrieval_queries": ["gap query"],
            "sub_question_context_counts": [{"question": "Q", "chunk_count": 0}],
            "diagnostics": {"retrieved_count": 0},
        }

        self.assertFalse(report_gap_refresh_has_context(refreshed))
        marked = mark_report_gap_refresh_empty(original, refreshed)

        self.assertEqual(marked["synthesis"], "existing synthesis")
        self.assertTrue(marked["diagnostics"]["existing"])
        self.assertTrue(marked["diagnostics"]["report_gap_retry"])
        self.assertTrue(marked["diagnostics"]["report_gap_refresh_empty"])
        self.assertEqual(marked["diagnostics"]["report_gap_retry_queries"], ["gap query"])


if __name__ == "__main__":
    unittest.main()

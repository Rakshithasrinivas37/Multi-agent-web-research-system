import unittest

from src.agents.planner_agent import (
    ResearchPlan,
    ResearchTask,
    build_sub_question_specs,
    clean_sub_questions,
)


class PlannerAgentTests(unittest.TestCase):
    def test_build_sub_question_specs_adds_stable_ids_and_required_evidence(self):
        specs = build_sub_question_specs([
            "What is the core equation and components?",
            "What benchmark results demonstrate performance?",
        ])

        self.assertEqual(specs[0].question_id, "q001")
        self.assertEqual(specs[1].question_id, "q002")
        self.assertIn("equation", specs[0].required_evidence)
        self.assertIn("benchmark", specs[1].required_evidence)

    def test_clean_sub_questions_accepts_legacy_strings_and_structured_items(self):
        questions = clean_sub_questions([
            "What is attention?",
            {"question": "How does it perform?", "required_evidence": ["benchmark"]},
        ])

        self.assertEqual(questions, ["What is attention?", "How does it perform?"])

    def test_research_plan_to_dict_preserves_plain_questions_and_specs(self):
        questions = ["What is the API usage?"]
        plan = ResearchPlan(
            objective="Test objective",
            research_mode="technical_deep_dive",
            sub_questions=questions,
            sub_question_specs=build_sub_question_specs(questions),
            tasks=[
                ResearchTask(
                    task_id="task_001",
                    query_context="Find docs",
                    url="SEARCH:official docs",
                    source_type="search",
                    priority=1,
                    extraction_goal="Find API usage.",
                )
            ],
            synthesis_instruction="Synthesize findings.",
            output_format="report",
        )

        data = plan.to_dict()

        self.assertEqual(data["sub_questions"], questions)
        self.assertEqual(data["sub_question_specs"][0]["question_id"], "q001")
        self.assertIn("api", data["sub_question_specs"][0]["required_evidence"])


if __name__ == "__main__":
    unittest.main()

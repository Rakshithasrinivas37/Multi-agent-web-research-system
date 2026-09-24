import unittest

from src.agents.planner_agent import (
    PlannerAgent,
    ResearchPlan,
    ResearchTask,
    apply_authoritative_search_hints,
    build_sub_question_specs,
    clean_sub_questions,
    ensure_sub_question_task_coverage,
    is_disallowed_source_url,
    preferred_candidates,
    validate_plan,
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
            "What is the concept?",
            {"question": "How does it perform?", "required_evidence": ["benchmark"]},
        ])

        self.assertEqual(questions, ["What is the concept?", "How does it perform?"])

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

    def test_authoritative_search_hints_add_primary_source_terms(self):
        tasks = [
            ResearchTask(
                task_id="task_001",
                query_context="What is the core equation and computational complexity?",
                url="SEARCH:core equation computational complexity",
                source_type="search",
                priority=1,
                extraction_goal="Extract equation, components, and complexity.",
            )
        ]

        result = apply_authoritative_search_hints(tasks, "technical_deep_dive")

        self.assertIn("original", result[0].url)
        self.assertIn("paper", result[0].url)
        self.assertIn("arxiv", result[0].url)
        self.assertIn("doi", result[0].url)

    def test_authoritative_search_hints_add_official_docs_terms(self):
        tasks = [
            ResearchTask(
                task_id="task_001",
                query_context="Find framework API usage.",
                url="SEARCH:framework API usage",
                source_type="search",
                priority=1,
                extraction_goal="Extract official API signatures and code usage.",
            )
        ]

        result = apply_authoritative_search_hints(tasks, "technical_deep_dive")

        url = result[0].url.lower()

        self.assertIn("official", url)
        self.assertIn("docs", url)
        self.assertIn("api", url)

    def test_video_and_social_sources_are_disallowed(self):
        self.assertTrue(is_disallowed_source_url("https://www.youtube.com/watch?v=test"))
        self.assertTrue(is_disallowed_source_url("https://youtu.be/test"))
        self.assertFalse(is_disallowed_source_url("https://arxiv.org/abs/1706.03762"))

    def test_validate_plan_rejects_disallowed_source_urls(self):
        tasks = [
            ResearchTask(
                task_id="task_001",
                query_context="Find primary evidence",
                url="https://www.youtube.com/watch?v=test",
                source_type="webpage",
                priority=1,
                extraction_goal="Extract evidence.",
            ),
            ResearchTask(
                task_id="task_002",
                query_context="Find official docs",
                url="SEARCH:official docs",
                source_type="search",
                priority=2,
                extraction_goal="Extract evidence.",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "disallowed source URL"):
            validate_plan(tasks, "knowledge_research", [], ["What is the evidence?"])

    def test_sub_question_coverage_adds_missing_task(self):
        questions = [
            "What is the definition of the method?",
            "What equation defines the method?",
        ]
        tasks = [
            ResearchTask(
                task_id="task_001",
                query_context=questions[0],
                url="SEARCH:method definition official source",
                source_type="search",
                priority=1,
                extraction_goal="Extract definition.",
            )
        ]

        result = ensure_sub_question_task_coverage(tasks, questions, "technical_deep_dive")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].query_context, questions[1])
        self.assertIn("equation", result[1].expected_signals)

    def test_secondary_technical_direct_url_becomes_authoritative_search(self):
        task = ResearchTask(
            task_id="task_001",
            query_context="What is the core equation?",
            url="https://example.com/tutorial/core-equation",
            source_type="technical_overview",
            priority=1,
            extraction_goal="Extract equation and components.",
        )

        result = PlannerAgent(use_llm=False, validate_urls=False)._safe_task(
            task,
            "Example method",
            "technical_deep_dive",
        )

        self.assertTrue(result.url.startswith("SEARCH:"))
        self.assertIn("original", result.url)
        self.assertIn("paper", result.url)

    def test_primary_technical_direct_urls_are_kept(self):
        task = ResearchTask(
            task_id="task_001",
            query_context="What is the core equation?",
            url="https://arxiv.org/pdf/1706.03762",
            source_type="arxiv",
            priority=1,
            extraction_goal="Extract equation and components.",
        )

        result = PlannerAgent(use_llm=False, validate_urls=False)._safe_task(
            task,
            "Example method",
            "technical_deep_dive",
        )

        self.assertEqual(result.url, "https://arxiv.org/pdf/1706.03762")

    def test_preferred_candidates_filter_to_authoritative_technical_sources(self):
        task = ResearchTask(
            task_id="task_001",
            query_context="What equation defines the method?",
            url="SEARCH:method equation",
            source_type="search",
            priority=1,
            extraction_goal="Extract equation.",
        )
        candidates = [
            {"title": "Tutorial", "url": "https://example.com/tutorial", "snippet": "A tutorial."},
            {"title": "Paper", "url": "https://arxiv.org/abs/1706.03762", "snippet": "Original paper."},
        ]

        result = preferred_candidates(task, candidates)

        self.assertEqual([item["url"] for item in result], ["https://arxiv.org/abs/1706.03762"])


if __name__ == "__main__":
    unittest.main()

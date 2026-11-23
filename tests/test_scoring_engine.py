
import unittest

from src.evaluation.scoring_engine import ScoringEngine


def sample_parameters():
    return {
        "safety_topics": {
            "hazard_awareness": {
                "mentioned": True,
                "matched_keywords": ["unsafe"],
            },
            "ppe_discussion": {
                "mentioned": False,
                "matched_keywords": [],
            },
            "emergency_information": {
                "mentioned": False,
                "matched_keywords": [],
            },
            "injury_prevention": {
                "mentioned": True,
                "matched_keywords": ["hurt"],
            },
            "stop_work_authority": {
                "mentioned": True,
                "matched_keywords": ["stop them"],
            },
            "equipment_safety": {
                "mentioned": True,
                "matched_keywords": ["equipment"],
            },
        },
        "ppe_assessment": {
            "video_predictions_available": True,
            "predictions": {"Hardhat": 45, "NO-Hardhat": 8},
            "compliance_percentage": None,
            "limitation": "Counts are detections, not unique workers.",
        },
    }


class TestScoringEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ScoringEngine()

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.score("not a dictionary")

    def test_missing_topics_dictionary_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.score({})

    def test_returns_all_ten_dimensions(self):
        result = self.engine.score(sample_parameters())
        self.assertEqual(len(result["dimensions"]), 10)

    def test_overall_quality_score_is_not_invented(self):
        result = self.engine.score(sample_parameters())
        self.assertIsNone(result["overall_score"])

    def test_topic_coverage_is_separate_from_overall_score(self):
        result = self.engine.score(sample_parameters())
        self.assertAlmostEqual(
            result["topic_detection_coverage_percentage"],
            4 / 6 * 100,
            places=2,
        )
        self.assertIsNone(result["overall_score"])

    def test_unmatched_topic_is_unknown(self):
        result = self.engine.score(sample_parameters())
        self.assertEqual(
            result["dimensions"]["ppe_safety"]["status"], "unknown"
        )

    def test_positive_keyword_match_is_recorded_as_evidence(self):
        result = self.engine.score(sample_parameters())
        self.assertEqual(
            result["dimensions"]["hazard_awareness"]["status"],
            "evidence_detected",
        )

    def test_missing_topic_result_is_unknown(self):
        data = sample_parameters()
        del data["safety_topics"]["equipment_safety"]

        result = self.engine.score(data)

        self.assertEqual(
            result["dimensions"]["equipment_safety"]["status"], "unknown"
        )
        self.assertEqual(
            result["topic_detection_count"]["with_results"], 5
        )

    def test_invalid_topic_value_is_unknown(self):
        data = sample_parameters()
        data["safety_topics"]["hazard_awareness"]["mentioned"] = "yes"

        result = self.engine.score(data)

        self.assertEqual(
            result["dimensions"]["hazard_awareness"]["status"], "unknown"
        )

    def test_unmeasured_dimensions_remain_unknown(self):
        result = self.engine.score(sample_parameters())

        for key in (
            "communication_clarity",
            "worker_engagement",
            "demonstration_visual_support",
        ):
            self.assertEqual(result["dimensions"][key]["status"], "unknown")
            self.assertIsNone(result["dimensions"][key]["score"])

    def test_ppe_video_predictions_are_preserved(self):
        result = self.engine.score(sample_parameters())
        video = result["dimensions"]["ppe_safety"]["video_evidence"]

        self.assertTrue(video["available"])
        self.assertEqual(video["predictions"]["Hardhat"], 45)
        self.assertIsNone(video["compliance_percentage"])

    def test_no_topic_results_gives_unknown_coverage(self):
        data = sample_parameters()
        for topic in data["safety_topics"].values():
            topic["mentioned"] = None

        result = self.engine.score(data)

        self.assertIsNone(result["topic_detection_coverage_percentage"])
        self.assertEqual(
            result["topic_detection_count"]["with_results"], 0
        )


if __name__ == "__main__":
    unittest.main()


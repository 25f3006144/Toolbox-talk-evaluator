
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.report_generator import ReportGenerator


def sample_parameters():
    return {
        "module": "parameter_extraction",
        "transcript": (
            "If you see something unsafe, stop the work. "
            "Look out for each other."
        ),
        "meeting_duration": {
            "duration_seconds": 44.53
        },
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
                "matched_keywords": ["look out for each other"],
            },
            "stop_work_authority": {
                "mentioned": True,
                "matched_keywords": ["stop the work"],
            },
            "equipment_safety": {
                "mentioned": False,
                "matched_keywords": [],
            },
        },
        "computer_vision": {
            "frames_analyzed": 23,
            "frames_with_people": 22,
            "frames_with_visible_faces": 21,
            "frames_with_possible_group_gathering": 20,
            "limitations": [
                "Face visibility does not prove attention."
            ],
        },
        "ppe_assessment": {
            "speech_mentioned": False,
            "video_predictions_available": True,
            "predictions": {
                "Hardhat": 45,
                "NO-Hardhat": 8,
                "Safety Vest": 35,
            },
            "compliance_percentage": None,
            "limitation": (
                "Predictions do not represent verified unique workers."
            ),
        },
        "transcript_statistics": {
            "word_count": 14,
            "sentence_count": 2,
            "average_words_per_sentence": 7,
            "timestamped_segment_count": 2,
            "clarity_assessment": "Not assessed.",
        },
        "evidence_limitations": [
            "Keyword matches do not prove understanding."
        ],
    }


def sample_scoring():
    return {
        "module": "scoring_engine",
        "overall_score": None,
        "score_scale": 100,
        "score_label": (
            "Unavailable: overall quality scoring rules are not yet validated"
        ),
        "topic_detection_coverage_percentage": 50.0,
        "topic_detection_count": {
            "detected": 3,
            "with_results": 6,
            "total_expected": 6,
        },
        "dimensions": {
            "hazard_awareness": {
                "name": "Hazard awareness",
                "score": 10,
                "max_score": 10,
                "status": "evidence_detected",
                "evidence": "Keyword unsafe matched.",
            },
            "ppe_safety": {
                "name": "PPE safety",
                "score": None,
                "max_score": 10,
                "status": "unknown",
                "evidence": "No keyword match.",
            },
            "safety_topic_coverage": {
                "name": "Safety topic coverage",
                "score": 50.0,
                "max_score": 100,
                "status": "measured",
                "evidence": "3 of 6 topics matched.",
            },
        },
        "limitations": [
            "PPE compliance cannot be calculated from predictions alone."
        ],
    }


class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ReportGenerator()
        self.parameters = sample_parameters()
        self.scoring = sample_scoring()
        self.report = self.generator.generate_report(
            self.parameters,
            self.scoring,
        )

    def test_report_is_markdown_text(self):
        self.assertIsInstance(self.report, str)
        self.assertIn("# Toolbox Talk Evaluation Report", self.report)

    def test_report_contains_all_main_sections(self):
        for heading in (
            "Executive summary",
            "Safety topic findings",
            "Scoring summary",
            "Ten-dimension assessment",
            "Computer vision and PPE observations",
            "Transcript and communication evidence",
            "Evidence-supported strengths",
            "Areas for review",
            "Limitations and responsible use",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.report)

    
    def test_overall_score_is_not_invented(self):
        self.assertIn(
            "**Overall quality score:** Not available",
            self.report,
        )
        self.assertIn(
            "overall quality scoring rules are not yet validated",
            self.report,
        )

    
    def test_topic_coverage_is_reported_separately(self):
        self.assertIn(
            "**Topic detection coverage:** 50.0%",
            self.report,
        )
        self.assertIn(
            "**Topics detected:** 3 of 6",
            self.report,
        )


    def test_detected_topic_appears_in_report(self):
        self.assertIn("Hazard awareness", self.report)
        self.assertIn("Evidence detected", self.report)
        self.assertIn("unsafe", self.report)

    def test_no_keyword_match_is_not_called_a_failure(self):
        self.assertIn("Unknown — no keyword match", self.report)
        self.assertNotIn("PPE safety | Failed", self.report)

    def test_ppe_predictions_are_included(self):
        self.assertIn("Hardhat", self.report)
        self.assertIn("45", self.report)
        self.assertIn("NO-Hardhat", self.report)

    
    def test_ppe_compliance_remains_unavailable(self):
        self.assertIn( 
            "**PPE compliance percentage:** Not calculated",
            self.report,
        )


    def test_visual_limitations_are_included(self):
        self.assertIn("Face visibility does not prove attention.", self.report)
        self.assertIn(
            "Predictions do not represent verified unique workers.",
            self.report,
        )

    def test_transcript_is_included(self):
        self.assertIn(
            "If you see something unsafe, stop the work.",
            self.report,
        )
        self.assertIn("Word count", self.report)

    def test_empty_transcript_is_supported(self):
        self.parameters["transcript"] = ""
        report = self.generator.generate_report(
            self.parameters,
            self.scoring,
        )
        self.assertIn(
            "No transcript text was available",
            report,
        )

    def test_invalid_input_types_are_rejected(self):
        with self.assertRaises(ValueError):
            self.generator.generate_report([], self.scoring)

        with self.assertRaises(ValueError):
            self.generator.generate_report(self.parameters, [])

    def test_report_is_saved_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parameter_path = root / "parameters.json"
            scoring_path = root / "scoring.json"
            output_path = root / "output" / "report.md"

            parameter_path.write_text(
                json.dumps(self.parameters),
                encoding="utf-8",
            )
            scoring_path.write_text(
                json.dumps(self.scoring),
                encoding="utf-8",
            )

            saved_path = self.generator.generate_from_files(
                parameter_path,
                scoring_path,
                output_path,
            )

            self.assertTrue(saved_path.is_file())
            content = saved_path.read_text(encoding="utf-8")
            self.assertIn(
                "# Toolbox Talk Evaluation Report",
                content,
            )

    def test_missing_input_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"

            with self.assertRaises(FileNotFoundError):
                self.generator.generate_from_files(
                    missing_path,
                    missing_path,
                    Path(temp_dir) / "report.md",
                )

    def test_invalid_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_json = root / "bad.json"
            good_json = root / "good.json"

            bad_json.write_text("{invalid json", encoding="utf-8")
            good_json.write_text(
                json.dumps(self.scoring),
                encoding="utf-8",
            )

            with self.assertRaises(json.JSONDecodeError):
                self.generator.generate_from_files(
                    bad_json,
                    good_json,
                    root / "report.md",
                )


if __name__ == "__main__":
    unittest.main()

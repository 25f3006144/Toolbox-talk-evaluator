import unittest

from src.evaluation.parameter_extractor import ParameterExtractor
from src.evaluation.parameter_validation import validate_parameter_output


class TestParameterExtractionAudit(unittest.TestCase):

    def setUp(self):
        self.extractor = ParameterExtractor()

    def test_safety_keywords_are_extracted(self):
        result = self.extractor.extract_parameters(
            transcript="Wear your hardhat. Stop work if you see a hazard."
        )

        keywords = result["safety_keywords"]["categories"]

        self.assertIn("hardhat", keywords["ppe"]["keywords"])
        self.assertIn("hazard", keywords["hazards"]["keywords"])
        self.assertIn("stop work", keywords["stop_work"]["keywords"])

    def test_language_clarity_does_not_invent_score(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe. Watch for hazards."
        )

        clarity = result["language_clarity"]

        self.assertTrue(clarity["transcript_available"])
        self.assertEqual(clarity["sentence_count"], 2)
        self.assertIsNone(clarity["clarity_score"])

    def test_empty_transcript_is_supported(self):
        result = self.extractor.extract_parameters(transcript="")

        self.assertFalse(
            result["language_clarity"]["transcript_available"]
        )
        self.assertIsNone(
            result["language_clarity"]["average_words_per_sentence"]
        )

    def test_unknown_video_signals_remain_unknown(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe.",
            cv_results={},
            metadata={"duration_seconds": 10},
        )

        signals = result["additional_signals"]

        self.assertIsNone(signals["attendance_percentage"])
        self.assertIsNone(signals["attention_level"])
        self.assertIsNone(signals["participation_level"])
        self.assertIsNone(signals["demonstration_detected"])
        self.assertEqual(signals["talk_duration_seconds"], 10)

    def test_optional_metadata_is_not_guessed(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe."
        )

        signals = result["additional_signals"]

        self.assertIsNone(signals["time_of_day"])
        self.assertIsNone(signals["supervisor_presence"])
        self.assertIsNone(signals["visual_aids_or_boards"])

    def test_validator_requires_new_sections(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe."
        )

        self.assertEqual(
            validate_parameter_output(result),
            [],
        )

        del result["safety_keywords"]

        errors = validate_parameter_output(result)

        self.assertTrue(
            any("safety_keywords" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
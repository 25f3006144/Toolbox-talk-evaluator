import unittest

from src.evaluation.parameter_extractor import ParameterExtractor
from src.evaluation.parameter_validation import validate_parameter_output


class TestParameterExtractionPart2(unittest.TestCase):

    def setUp(self):
        self.extractor = ParameterExtractor()

    def test_extracts_safety_recommendation(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe and look out for each other."
        )

        instructions = result["safety_instructions"]

        self.assertGreater(instructions["count"], 0)
        self.assertTrue(
            any(
                "be safe" in item["text"].lower()
                for item in instructions["items"]
            )
        )

    def test_extracts_wear_instruction(self):
        result = self.extractor.extract_parameters(
            transcript="Wear your hardhat before entering the site."
        )

        instructions = result["safety_instructions"]

        self.assertTrue(
            any(
                "wear your hardhat" in item["text"].lower()
                for item in instructions["items"]
            )
        )

    def test_instruction_count_matches_items(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe. Report hazards. Look out for each other."
        )

        instructions = result["safety_instructions"]

        self.assertEqual(
            instructions["count"],
            len(instructions["items"]),
        )

    def test_ppe_percentage_remains_unknown(self):
        result = self.extractor.extract_parameters(
            transcript="Wear your safety vest."
        )

        self.assertIsNone(
            result["ppe_assessment"]["compliance_percentage"]
        )

    def test_output_has_required_part_two_fields(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe."
        )

        self.assertIn("safety_instructions", result)
        self.assertIn("ppe_assessment", result)

    def test_validates_correct_output(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe. Report hazards."
        )

        errors = validate_parameter_output(result)

        self.assertEqual(errors, [])

    def test_validator_detects_missing_field(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe."
        )
        del result["ppe_assessment"]

        errors = validate_parameter_output(result)

        self.assertTrue(
            any("ppe_assessment" in error for error in errors)
        )

    def test_validator_detects_instruction_count_mismatch(self):
        result = self.extractor.extract_parameters(
            transcript="Be safe."
        )
        result["safety_instructions"]["count"] = 99

        errors = validate_parameter_output(result)

        self.assertTrue(
            any("instruction count" in error.lower() for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
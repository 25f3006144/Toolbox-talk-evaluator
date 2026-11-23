
import unittest

from src.evaluation.parameter_extractor import ParameterExtractor


class TestParameterExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = ParameterExtractor()

    def test_extracts_ppe_and_emergency_topics(self):
        transcript = (
            "Wear your hard hat and safety vest. "
            "Know the emergency exit."
        )

        results = self.extractor.extract_parameters(transcript)

        topics = results["safety_topics"]

        self.assertTrue(topics["ppe_discussion"]["mentioned"])
        self.assertTrue(
            topics["emergency_information"]["mentioned"]
        )

    def test_topic_not_mentioned(self):
        transcript = "Good morning, everyone."

        results = self.extractor.extract_parameters(transcript)

        topics = results["safety_topics"]

        self.assertFalse(topics["emergency_information"]["mentioned"])

    def test_counts_words_and_sentences(self):
        transcript = "Wear your helmet. Stay safe."

        results = self.extractor.extract_parameters(transcript)

        statistics = results["transcript_statistics"]

        self.assertEqual(statistics["word_count"], 5)
        self.assertEqual(statistics["sentence_count"], 2)

    def test_detects_question_mark(self):
        transcript = "Where is the emergency exit?"

        results = self.extractor.extract_parameters(transcript)

        self.assertEqual(results["questions"]["count"], 1)
        
    def test_reads_transcriber_json_format(self):
        transcript = {
            "language": "en",
            "language_probability": 0.99,
            "transcript": "Wear your hard hat. Where is the emergency exit?",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Wear your hard hat."
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Where is the emergency exit?"
                }
            ]
        }

        results = self.extractor.extract_parameters(transcript)

        self.assertTrue(results["transcript"]["available"])
        self.assertEqual(
            results["transcript"]["text"],
            transcript["transcript"]
        )
        self.assertEqual(results["questions"]["count"], 1)
        self.assertEqual(
            results["questions"]["items"][0]["start"],
            2.0
        )
        self.assertTrue(
            results["safety_topics"]["ppe_discussion"]["mentioned"]
        )



    def test_reads_timestamped_segments(self):
        transcript = {
            "text": "What should we do?",
            "segments": [
                {
                    "start": 1.0,
                    "end": 2.5,
                    "text": "What should we do?"
                }
            ],
        }

        results = self.extractor.extract_parameters(transcript)

        self.assertEqual(results["questions"]["count"], 1)
        self.assertEqual(
            results["questions"]["items"][0]["start"],
            1.0,
        )

    def test_uses_metadata_duration(self):
        results = self.extractor.extract_parameters(
            "Wear your hard hat.",
            metadata={"duration_seconds": 45.5},
        )

        duration = results["meeting_duration"]

        self.assertEqual(duration["seconds"], 45.5)
        self.assertTrue(duration["is_full_meeting_duration"])

    def test_missing_duration_stays_unknown(self):
        results = self.extractor.extract_parameters(
            "Stay safe."
        )

        self.assertIsNone(results["meeting_duration"]["seconds"])

    def test_records_evidence_limitations(self):
        results = self.extractor.extract_parameters("Stay safe.")

        self.assertGreater(
            len(results["evidence_limitations"]),
            0,
        )

    def test_empty_transcript_does_not_crash(self):
        results = self.extractor.extract_parameters("")

        self.assertFalse(results["transcript"]["available"])
        self.assertEqual(results["transcript"]["word_count"], 0)
    def test_reads_nested_cv_pipeline_results(self):
        extractor = ParameterExtractor()

        cv_results = {
            "person_detection": [
                {"timestamp_seconds": 0.0, "people_count": 0},
                {"timestamp_seconds": 2.0, "people_count": 1},
                {"timestamp_seconds": 4.0, "people_count": 3},
            ],
            "scene_analysis": {
                "summary": {
                    "frames_analyzed": 3,
                    "ppe_detections": {
                        "Hardhat": 5,
                        "Safety Vest": 3,
                    },
                }
            },
            "activity_recognition": {
                "summary": {
                    "frames_with_visible_faces": 2,
                    "frames_with_possible_group_gathering": 2,
                }
            },
            "limitations": ["Test limitation"],
        }

        result = extractor.extract_parameters(
            transcript="Workers must wear hard hats.",
            cv_results=cv_results,
            metadata={"duration_seconds": 6.0},
        )

        cv_summary = result["computer_vision"]

        self.assertEqual(cv_summary["total_frames"], 3)
        self.assertEqual(cv_summary["frames_with_people"], 2)
        self.assertEqual(cv_summary["frames_with_faces"], 2)
        self.assertEqual(cv_summary["possible_group_gathering"], 2)
        self.assertEqual(
            cv_summary["ppe_predictions"]["Hardhat"], 5
        )
        self.assertIsNone(cv_summary["attendance_estimate"]["value"])
        self.assertIsNone(cv_summary["attention"]["value"])   
    def test_detects_stop_work_authority(self):
        extractor = ParameterExtractor()

        result = extractor.extract_parameters(
            transcript="Everybody here has the right to stop them to save their life.",
        )

        topic = result["safety_topics"]["stop_work_authority"]

        self.assertTrue(topic["mentioned"])
        self.assertIn("right to stop", topic["matched_keywords"])   
    def test_statement_starting_with_what_is_not_a_question(self):
        extractor = ParameterExtractor()

        text = (
            "What we can't fix is if one of you guys gets hurt, "
            "you cannot go back on that."
        )

        self.assertFalse(extractor._looks_like_question(text))                 

    def test_detects_all_six_safety_topics(self):
        extractor = ParameterExtractor()

        transcript = (
            "Watch out for hazards and unsafe conditions. "
            "Wear your hardhat and safety vest. "
            "Know the emergency exit and first aid procedure. "
            "Prevent injuries and help workers go home safely. "
            "Everyone has the right to stop unsafe work. "
            "Keep clear of moving equipment and machinery."
        )

        result = extractor.extract_parameters(
            transcript=transcript,
        )

        expected_topics = [
            "hazard_awareness",
            "ppe_discussion",
            "emergency_information",
            "injury_prevention",
            "stop_work_authority",
            "equipment_safety",
        ]

        for topic_name in expected_topics:
            with self.subTest(topic=topic_name):
                self.assertTrue(
                    result["safety_topics"][topic_name]["mentioned"],
                    f"{topic_name} should have been detected",
                )

    def test_does_not_mark_unmentioned_topics(self):
        extractor = ParameterExtractor()

        result = extractor.extract_parameters(
            transcript="Welcome everyone. Thank you for attending today's meeting."
        )

        for topic_name, topic_result in result["safety_topics"].items():
            with self.subTest(topic=topic_name):
                self.assertFalse(topic_result["mentioned"])

def test_detects_ppe_topic_when_mentioned(self):
    result = self.extractor.extract_parameters(
        transcript="Workers must wear a hardhat and safety vest."
    )
    self.assertTrue(
        result["safety_topics"]["ppe_discussion"]["mentioned"]
    )

def test_does_not_detect_emergency_topic_when_absent(self):
    result = self.extractor.extract_parameters(
        transcript="Look out for unsafe equipment."
    )
    self.assertFalse(
        result["safety_topics"]["emergency_information"]["mentioned"]
    )                


                
if __name__ == "__main__":
    unittest.main()

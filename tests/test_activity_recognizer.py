
import unittest

from src.cv.activity_recognizer import ActivityRecognizer


class TestActivityRecognizer(unittest.TestCase):

    def test_group_gathering_detected(self):
        cues = ActivityRecognizer.classify_activity_cues(
            person_count=5,
            face_count=2
        )

        self.assertIn("possible_group_gathering", cues)
        self.assertIn("faces_visible", cues)

    def test_no_clear_cue(self):
        cues = ActivityRecognizer.classify_activity_cues(
            person_count=0,
            face_count=0
        )

        self.assertEqual(
            cues,
            ["no_clear_activity_cue"]
        )

    def test_summary_counts_frames(self):
        frame_results = [
            {
                "people_count": 4,
                "face_count": 2,
                "activity_cues": [
                    "possible_group_gathering",
                    "faces_visible"
                ]
            },
            {
                "people_count": 1,
                "face_count": 0,
                "activity_cues": [
                    "no_clear_activity_cue"
                ]
            },
            {
                "people_count": 3,
                "face_count": 1,
                "activity_cues": [
                    "possible_group_gathering",
                    "faces_visible"
                ]
            }
        ]

        summary = ActivityRecognizer.summarize_results(
            frame_results
        )

        self.assertEqual(summary["frames_analyzed"], 3)
        self.assertEqual(summary["frames_with_people"], 3)
        self.assertEqual(
            summary["frames_with_visible_faces"], 2
        )
        self.assertEqual(
            summary["frames_with_possible_group_gathering"],
            2
        )

    def test_empty_summary(self):
        summary = ActivityRecognizer.summarize_results([])

        self.assertEqual(summary["frames_analyzed"], 0)
        self.assertEqual(summary["frames_with_people"], 0)
        self.assertEqual(
            summary["frames_with_visible_faces"], 0
        )


if __name__ == "__main__":
    unittest.main()

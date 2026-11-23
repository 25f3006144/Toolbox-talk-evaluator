
import unittest

from src.cv.scene_analyzer import SceneAnalyzer


class TestSceneAnalyzer(unittest.TestCase):

    def test_empty_results(self):
        summary = SceneAnalyzer.summarize_results([])

        self.assertEqual(summary["frames_analyzed"], 0)
        self.assertEqual(summary["frames_with_person"], 0)
        self.assertEqual(summary["detection_counts"], {})

    def test_summarizes_scene_detections(self):
        sample_results = [
            {
                "timestamp_seconds": 0.0,
                "detections": [],
            },
            {
                "timestamp_seconds": 2.0,
                "detections": [
                    {"label": "Person", "confidence": 0.91},
                    {"label": "Hardhat", "confidence": 0.80},
                    {"label": "Safety Vest", "confidence": 0.85},
                    {"label": "machinery", "confidence": 0.76},
                ],
            },
            {
                "timestamp_seconds": 4.0,
                "detections": [
                    {"label": "Person", "confidence": 0.88},
                    {"label": "NO-Hardhat", "confidence": 0.65},
                    {"label": "Safety Cone", "confidence": 0.72},
                ],
            },
        ]

        summary = SceneAnalyzer.summarize_results(
            sample_results
        )

        self.assertEqual(summary["frames_analyzed"], 3)
        self.assertEqual(summary["frames_with_person"], 2)
        self.assertEqual(
            summary["frames_with_environment_objects"], 2
        )
        self.assertEqual(
            summary["detection_counts"]["Person"], 2
        )
        self.assertEqual(
            summary["environment_objects"]["machinery"], 1
        )
        self.assertEqual(
            summary["environment_objects"]["Safety Cone"], 1
        )
        self.assertEqual(
            summary["ppe_detections"]["NO-Hardhat"], 1
        )

    def test_counts_are_not_unique_people(self):
        sample_results = [
            {
                "detections": [
                    {"label": "Person", "confidence": 0.90},
                ]
            },
            {
                "detections": [
                    {"label": "Person", "confidence": 0.92},
                ]
            },
        ]

        summary = SceneAnalyzer.summarize_results(
            sample_results
        )

        self.assertEqual(
            summary["detection_counts"]["Person"], 2
        )
        self.assertEqual(summary["frames_with_person"], 2)

    def test_missing_ppe_is_noted_as_a_prediction(self):
        summary = SceneAnalyzer.summarize_results([
            {
                "detections": [
                    {"label": "NO-Mask", "confidence": 0.70},
                ]
            }
        ])

        self.assertIn(
            "NO-Mask", summary["ppe_detections"]
        )
        self.assertTrue(
            any("visually verified" in note
                for note in summary["notes"])
        )


if __name__ == "__main__":
    unittest.main()

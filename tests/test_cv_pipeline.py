
import unittest
from pathlib import Path

from src.evaluation.cv_pipeline import CVPipeline


class TestCVPipeline(unittest.TestCase):

    def test_pipeline_initializes(self):
        pipeline = CVPipeline()

        self.assertIsNotNone(pipeline.person_detector)
        self.assertIsNotNone(pipeline.attention_detector)
        self.assertIsNotNone(pipeline.ppe_detector)
        self.assertIsNotNone(pipeline.scene_analyzer)
        self.assertIsNotNone(pipeline.activity_recognizer)

    def test_missing_video_raises_error(self):
        pipeline = CVPipeline()

        with self.assertRaises(FileNotFoundError):
            pipeline.analyze_video(
                "data/input/missing_video.mp4"
            )

    def test_invalid_sample_interval_raises_error(self):
        pipeline = CVPipeline()

        with self.assertRaises(ValueError):
            pipeline.analyze_video(
                "data/input/con.mp4",
                sample_interval=0
            )


if __name__ == "__main__":
    unittest.main()

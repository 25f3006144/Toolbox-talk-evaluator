
import unittest
from pathlib import Path

import cv2

from src.cv.ppe_detector import PPEDetector


VIDEO_PATH = Path("data/input/con.mp4")


class TestPPEDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the model only once for the test class.
        cls.detector = PPEDetector(
            confidence_threshold=0.35
        )

    def test_model_initializes(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.class_names)

        print("Model classes:", self.detector.class_names)

    def test_detect_ppe_in_frame(self):
        self.assertTrue(
            VIDEO_PATH.is_file(),
            f"Video not found: {VIDEO_PATH}"
        )

        cap = cv2.VideoCapture(str(VIDEO_PATH))

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            success, frame = cap.read()

            self.assertTrue(
                success,
                "Could not read the video frame."
            )

            detections = self.detector.detect_ppe(frame)

            self.assertIsInstance(detections, list)

            for detection in detections:
                self.assertIn("label", detection)
                self.assertIn("confidence", detection)
                self.assertIn("bbox", detection)

                self.assertGreaterEqual(
                    detection["confidence"], 0.35
                )
                self.assertLessEqual(
                    detection["confidence"], 1.0
                )
                self.assertEqual(
                    len(detection["bbox"]), 4
                )

            print("PPE detections:", detections)

        finally:
            cap.release()

    def test_draw_detections(self):
        import numpy as np

        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        detections = [{
            "label": "Hardhat",
            "confidence": 0.90,
            "bbox": [20, 20, 100, 100],
        }]

        output = self.detector.draw_detections(
            frame, detections
        )

        self.assertEqual(output.shape, frame.shape)
        self.assertIsNot(output, frame)

    def test_invalid_sample_interval(self):
        with self.assertRaises(ValueError):
            self.detector.analyze_video(
                VIDEO_PATH, sample_interval=0
            )


if __name__ == "__main__":
    unittest.main()

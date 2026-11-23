import unittest
import cv2
from pathlib import Path

from src.cv.attention_detector import AttentionDetector


class TestAttentionDetector(unittest.TestCase):

    def test_attention_detector_initialize(self):
        detector = AttentionDetector()
        self.assertIsNotNone(detector.face_detector)

    def test_detect_faces_in_video_frame(self):
        video_path = Path("data/input/con.mp4")
        self.assertTrue(
            video_path.exists(),
            f"Video not found: {video_path}"
        )

        detector = AttentionDetector()
        cap = cv2.VideoCapture(str(video_path))

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            success, frame = cap.read()

            self.assertTrue(success, "Could not read video frame")

            faces = detector.detect_faces(frame)

            print("Faces detected:", len(faces))
            self.assertIsNotNone(faces)

        finally:
            cap.release()

    def test_estimate_head_pose(self):
        from src.cv.attention_detector import AttentionDetector
        import cv2
        from pathlib import Path

        video_path = Path("data/input/con.mp4")
        detector = AttentionDetector()
        cap = cv2.VideoCapture(str(video_path))

        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            success, frame = cap.read()
            self.assertTrue(success, "Could not read video frame")

            faces = detector.detect_faces(frame)
            self.assertGreater(len(faces), 0, "No faces detected in test frame")

            pose = detector.estimate_head_pose(frame, faces[0])

            self.assertIsNotNone(pose, "Head-pose estimation failed")
            self.assertIn("pitch", pose)
            self.assertIn("yaw", pose)
            self.assertIn("roll", pose)

            print("Estimated head pose:", pose)
            self.assertTrue(all(
                isinstance(angle, float)
                for angle in pose.values()
            ))
            self.assertTrue(all(
                -180.0 <= angle <= 180.0
                for angle in pose.values()
            ))
             

        finally:
            cap.release()             


if __name__ == "__main__":
    unittest.main()
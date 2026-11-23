
import cv2
from pathlib import Path


class AttentionDetector:
    """Detect visible faces using OpenCV's YuNet face detector."""

    def __init__(self):
        self.model_path = (
            Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face detection model not found: {self.model_path}"
            )

        self.face_detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            0.7,
            0.3,
            5000
        )

    def detect_faces(self, frame):
        """Return detected faces in a video frame."""

        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))

        _, faces = self.face_detector.detect(frame)

        if faces is None:
            return []

        return faces   
      
    
    def estimate_head_pose(self, frame, face):
        """Estimate approximate head orientation from YuNet landmarks."""

        import numpy as np

        # YuNet landmarks: right eye, left eye, nose,
        # right mouth corner, left mouth corner.
        right_eye = np.array([face[4], face[5]], dtype=float)
        left_eye = np.array([face[6], face[7]], dtype=float)
        nose = np.array([face[8], face[9]], dtype=float)
        right_mouth = np.array([face[10], face[11]], dtype=float)
        left_mouth = np.array([face[12], face[13]], dtype=float)

        # Estimate roll from the eye-line angle.
        eye_line = left_eye - right_eye
        roll = np.degrees(
            np.arctan2(eye_line[1], eye_line[0])
        )

        # Estimate left/right turning using nose position
        # relative to the midpoint between the eyes.
        eye_midpoint = (right_eye + left_eye) / 2
        eye_distance = np.linalg.norm(left_eye - right_eye)

        if eye_distance < 1:
            return None

        nose_offset = (nose[0] - eye_midpoint[0]) / eye_distance
        yaw = float(np.degrees(np.arctan(nose_offset)))

        # Estimate up/down direction from nose position
        # relative to the eyes and mouth.
        mouth_midpoint = (right_mouth + left_mouth) / 2
        eye_to_mouth = mouth_midpoint[1] - eye_midpoint[1]

        if abs(eye_to_mouth) < 1:
            return None

        nose_position = (
            (nose[1] - eye_midpoint[1]) / eye_to_mouth
        )
        pitch = float(
            np.degrees(np.arctan(nose_position - 0.5))
        )

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": float(roll)
        }

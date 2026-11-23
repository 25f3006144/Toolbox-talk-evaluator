
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class PPEDetector:
    """Detect construction PPE using a pretrained YOLO model."""

    def __init__(self, confidence_threshold=0.35):
        if not 0 < confidence_threshold < 1:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        self.confidence_threshold = confidence_threshold

        # Download model weights into the Hugging Face cache
        # if they are not already available locally.
        model_path = hf_hub_download(
            repo_id="yihong1120/Construction-Hazard-Detection",
            filename="models/yolo11/pt/yolo11m.pt",
        )

        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_ppe(self, frame):
        """Return PPE-related detections for one image frame."""

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            label = self.class_names[class_id]

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

        return detections

    def draw_detections(self, frame, detections):
        """Draw detected object boxes and labels on a copy."""

        output = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = detection["label"]
            confidence = detection["confidence"]

            text = f"{label}: {confidence:.2f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return output

    def analyze_video(self, video_path, sample_interval=2.0):
        """Analyze a video at regular time intervals."""

        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than zero."
            )

        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        results = []

        try:
            duration_ms = (
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
                * 1000.0
                / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
            )

            timestamp_ms = 0.0

            while timestamp_ms < duration_ms:
                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp_ms,
                )

                success, frame = cap.read()

                if not success:
                    break

                detections = self.detect_ppe(frame)

                results.append({
                    "timestamp_seconds": timestamp_ms / 1000.0,
                    "detections": detections,
                })

                timestamp_ms += sample_interval * 1000.0

        finally:
            cap.release()

        return results

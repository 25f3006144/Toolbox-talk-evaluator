
from collections import Counter
from pathlib import Path

from src.cv.ppe_detector import PPEDetector


class SceneAnalyzer:
    """Summarize objects detected in sampled construction-site frames."""

    ENVIRONMENT_LABELS = {
        "machinery",
        "vehicle",
        "Safety Cone",
        "utility pole",
    }

    PPE_LABELS = {
        "Hardhat",
        "NO-Hardhat",
        "Safety Vest",
        "NO-Safety Vest",
        "Mask",
        "NO-Mask",
    }

    def __init__(self, confidence_threshold=0.50):
        self.detector = PPEDetector(
            confidence_threshold=confidence_threshold
        )

    @staticmethod
    def summarize_results(video_results):
        """Summarize frame-level detections without claiming unique counts."""

        if not video_results:
            return {
                "frames_analyzed": 0,
                "frames_with_person": 0,
                "frames_with_environment_objects": 0,
                "frames_with_ppe_detections": 0,
                "detection_counts": {},
                "environment_objects": [],
                "ppe_detections": {},
                "notes": [
                    "No frames were analyzed."
                ],
            }

        detection_counts = Counter()
        frames_with_person = 0
        frames_with_environment_objects = 0
        frames_with_ppe_detections = 0

        for frame_result in video_results:
            detections = frame_result.get("detections", [])

            labels_in_frame = {
                item.get("label")
                for item in detections
            }

            detection_counts.update(
                item.get("label", "Unknown")
                for item in detections
            )

            if "Person" in labels_in_frame:
                frames_with_person += 1

            if labels_in_frame & SceneAnalyzer.ENVIRONMENT_LABELS:
                frames_with_environment_objects += 1

            if labels_in_frame & SceneAnalyzer.PPE_LABELS:
                frames_with_ppe_detections += 1

        environment_objects = {
            label: detection_counts[label]
            for label in sorted(
                SceneAnalyzer.ENVIRONMENT_LABELS
            )
            if detection_counts[label] > 0
        }

        ppe_detections = {
            label: detection_counts[label]
            for label in sorted(SceneAnalyzer.PPE_LABELS)
            if detection_counts[label] > 0
        }

        return {
            "frames_analyzed": len(video_results),
            "frames_with_person": frames_with_person,
            "frames_with_environment_objects":
                frames_with_environment_objects,
            "frames_with_ppe_detections":
                frames_with_ppe_detections,
            "detection_counts": dict(
                sorted(detection_counts.items())
            ),
            "environment_objects": environment_objects,
            "ppe_detections": ppe_detections,
            "notes": [
                "Counts are detections across sampled frames, "
                "not unique people or unique objects.",
                "Missing-PPE labels are model predictions and "
                "must be visually verified.",
                "Object presence alone does not prove that a "
                "tool, vehicle, or machine is being used safely.",
            ],
        }

    def analyze_video(self, video_path, sample_interval=2.0):
        """Analyze a video and return its scene summary."""

        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        results = self.detector.analyze_video(
            video_path,
            sample_interval=sample_interval,
        )

        summary = self.summarize_results(results)

        return {
            "video_path": str(video_path),
            "sample_interval_seconds": sample_interval,
            "summary": summary,
        }

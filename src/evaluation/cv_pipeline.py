
from pathlib import Path

from src.cv.person_detector import PersonDetector
from src.cv.attention_detector import AttentionDetector
from src.cv.ppe_detector import PPEDetector
from src.cv.scene_analyzer import SceneAnalyzer
from src.cv.activity_recognizer import ActivityRecognizer


class CVPipeline:
    """Coordinate all computer vision modules for a toolbox-talk video."""

    def __init__(
        self,
        person_detector=None,
        attention_detector=None,
        ppe_detector=None,
        scene_analyzer=None,
        activity_recognizer=None,
    ):
        # Allow existing modules to be supplied, making testing easier.
        self.person_detector = (
            person_detector
            if person_detector is not None
            else PersonDetector()
        )

        self.attention_detector = (
            attention_detector
            if attention_detector is not None
            else AttentionDetector()
        )

        self.ppe_detector = (
            ppe_detector
            if ppe_detector is not None
            else PPEDetector()
        )

        self.scene_analyzer = (
            scene_analyzer
            if scene_analyzer is not None
            else SceneAnalyzer()
        )

        self.activity_recognizer = (
            activity_recognizer
            if activity_recognizer is not None
            else ActivityRecognizer(
                person_detector=self.person_detector,
                attention_detector=self.attention_detector,
            )
        )

    def analyze_video(self, video_path, sample_interval=2.0):
        """Run all CV analyses and return a combined result."""

        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than 0"
            )

        # 1. Person counts at sampled timestamps.
        person_results = self.person_detector.analyze_video(
            video_path,
            sample_interval=sample_interval,
        )

        # 2. PPE detections and scene summary.
        scene_results = self.scene_analyzer.analyze_video(
            video_path,
            sample_interval=sample_interval,
        )

        # 3. Activity cues, faces and head-pose estimates.
        activity_results = self.activity_recognizer.analyze_video(
            video_path,
            sample_interval=sample_interval,
        )

        # Keep the outputs separate and transparent.
        # These are not yet final compliance or attention scores.
        return {
            "video_path": str(video_path),
            "sample_interval_seconds": sample_interval,
            "person_detection": person_results,
            "scene_analysis": scene_results,
            "activity_recognition": activity_results,
            "limitations": [
                "People detected in different frames may be the same people.",
                "Face visibility and head direction do not prove attention.",
                "PPE predictions require visual verification.",
                "Activity cues do not prove speaking or participation.",
            ],
        }
 

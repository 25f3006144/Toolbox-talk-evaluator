
import cv2

from src.cv.person_detector import PersonDetector
from src.cv.attention_detector import AttentionDetector


class ActivityRecognizer:
    """
    Identify simple visual activity cues in toolbox-talk videos.

    This module uses person detection and face detection to estimate
    whether a group is gathered and whether faces are visible.
    It does not identify specific actions or prove that someone is
    listening or speaking.
    """

    def __init__(
        self,
        person_detector=None,
        attention_detector=None
    ):
        # Reuse our existing computer vision modules.
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

    @staticmethod
    def classify_activity_cues(person_count, face_count):
        """Classify simple activity clues from people and face counts."""

        cues = []

        if person_count >= 3:
            cues.append("possible_group_gathering")

        if face_count > 0:
            cues.append("faces_visible")

        if not cues:
            cues.append("no_clear_activity_cue")

        return cues

    def analyze_frame(self, frame):
        """Analyze one frame for people, visible faces and basic cues."""

        people = self.person_detector.detect_people(frame)
        faces = self.attention_detector.detect_faces(frame)

        head_poses = []

        for face in faces:
            pose = self.attention_detector.estimate_head_pose(
                frame, face
            )

            if pose is not None:
                head_poses.append(pose)

        person_count = len(people)
        face_count = len(faces)

        return {
            "people_count": person_count,
            "face_count": face_count,
            "head_pose_estimates": head_poses,
            "activity_cues": self.classify_activity_cues(
                person_count, face_count
            )
        }

    @staticmethod
    def summarize_results(frame_results):
        """Summarize activity cues across sampled video frames."""

        frames_analyzed = len(frame_results)

        frames_with_people = sum(
            1 for result in frame_results
            if result["people_count"] > 0
        )

        frames_with_faces = sum(
            1 for result in frame_results
            if result["face_count"] > 0
        )

        frames_with_group_gathering = sum(
            1 for result in frame_results
            if "possible_group_gathering"
            in result["activity_cues"]
        )

        return {
            "frames_analyzed": frames_analyzed,
            "frames_with_people": frames_with_people,
            "frames_with_visible_faces": frames_with_faces,
            "frames_with_possible_group_gathering":
                frames_with_group_gathering,
            "note": (
                "These are visual estimates from sampled frames. "
                "They do not prove attention, understanding, "
                "speaking, or participation."
            )
        }

    def analyze_video(self, video_path, sample_interval=2.0):
        """Analyze sampled frames throughout a video."""

        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than 0"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        frame_results = []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if fps <= 0 or total_frames <= 0:
                raise ValueError(
                    "Could not read video properties"
                )

            duration = total_frames / fps
            second = 0.0

            while second < duration:
                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    second * 1000
                )

                success, frame = cap.read()

                if not success:
                    second += sample_interval
                    continue

                result = self.analyze_frame(frame)

                result["timestamp_seconds"] = round(
                    second, 2
                )

                frame_results.append(result)
                second += sample_interval

        finally:
            cap.release()

        return {
            "video_path": str(video_path),
            "sample_interval_seconds": sample_interval,
            "frame_results": frame_results,
            "summary": self.summarize_results(frame_results)
        }

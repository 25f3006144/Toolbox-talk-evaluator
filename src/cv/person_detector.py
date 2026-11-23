
from ultralytics import YOLO


class PersonDetector:
    """Detect people in video frames using a pretrained YOLO model."""

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect_people(self, frame):
        """Detect people in a single video frame."""

        results = self.model(frame, conf=0.15, verbose=False)

        people = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                # COCO class 0 represents a person.
                if class_id == 0:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    people.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0])
                    })

        return people
    def draw_detections(self, frame, people):
        """Draw bounding boxes around detected people."""

        import cv2

        for person in people:
            x1, y1, x2, y2 = person["bbox"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person: {person['confidence']:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return frame
    def analyze_video(self, video_path, sample_interval=1.0):
        """Detect people in sampled frames throughout a video."""

        import cv2

        if sample_interval <= 0:
            raise ValueError("sample_interval must be greater than 0")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        results = []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or total_frames <= 0:
                raise ValueError("Could not read video properties")

            duration = total_frames / fps
            second = 0.0

            while second < duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                success, frame = cap.read()

                if not success:
                    second += sample_interval
                    continue

                people = self.detect_people(frame)

                results.append({
                    "timestamp_seconds": round(second, 2),
                    "people_count": len(people),
                    "people": people
                })

                second += sample_interval

        finally:
            cap.release()

        return results    
    

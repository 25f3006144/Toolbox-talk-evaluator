
import unittest

from src.cv.person_detector import PersonDetector


class TestPersonDetector(unittest.TestCase):

    def test_detector_initializes(self):
        detector = PersonDetector()

        self.assertIsNotNone(detector.model)

    def test_detect_people_in_video_frame(self):
        import cv2
        from pathlib import Path

        video_path = Path("data/input/con.mp4")
        self.assertTrue(video_path.exists(), f"Video not found: {video_path}")

        detector = PersonDetector()
        cap = cv2.VideoCapture(str(video_path))
        try:
            detected_any = False

            for _ in range(30):
                detected_any = False
                frames_checked = 0
                frame_results = []

            for second in range(0, 11, 2):
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                success, frame = cap.read()

                if not success:
                    continue

                frames_checked += 1
                people = detector.detect_people(frame)
                frame_results.append({
                        "timestamp_seconds": second,
                        "people_count": len(people)
                    })

                print(f"At {second}s: {len(people)} people detected")

                if people:
                    detected_any = True

            print(f"\nFrames checked: {frames_checked}")
            print(f"Person detected in video: {detected_any}")
            print("\nFrame results:")
            for result in frame_results:
                print(result)

            self.assertGreater(frames_checked, 0, "No video frames could be read")
            self.assertTrue(detected_any, "No people detected in sampled frames")
           

                

        finally:
            cap.release()

    def test_analyze_video(self):
        from pathlib import Path

        video_path = Path("data/input/con.mp4")
        self.assertTrue(
            video_path.exists(),
            f"Video not found: {video_path}"
        )

        detector = PersonDetector()
        results = detector.analyze_video(
            video_path,
            sample_interval=2.0
        )

        self.assertGreater(len(results), 0)

        print("\nFull-video analysis results:")

        for result in results:
            print(
                f"Time: {result['timestamp_seconds']}s, "
                f"People detected: {result['people_count']}"
            )            


        

            
    





if __name__ == "__main__":
    unittest.main()

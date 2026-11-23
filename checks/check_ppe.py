
import cv2

from src.cv.ppe_detector import PPEDetector


video_path = "data/input/con.mp4"

detector = PPEDetector(confidence_threshold=0.50)
cap = cv2.VideoCapture(video_path)

try:
    for seconds in [2, 4, 6, 8, 10]:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame at {seconds}s.")
            continue

        detections = detector.detect_ppe(frame)

        print(f"\nTime: {seconds}s")
        for detection in detections:
            print(detection)

        output = detector.draw_detections(
            frame, detections
        )

        cv2.imshow(f"PPE Detection - {seconds}s", output)
        cv2.waitKey(0)

finally:
    cap.release()
    cv2.destroyAllWindows()

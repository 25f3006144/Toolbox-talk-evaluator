
import cv2
from src.cv.attention_detector import AttentionDetector

video_path = "data/input/con.mp4"

detector = AttentionDetector()
cap = cv2.VideoCapture(video_path)

try:
    for seconds in [2, 4, 6]:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame at {seconds} seconds.")
            continue

        faces = detector.detect_faces(frame)
        print(f"\nTime: {seconds}s | Faces: {len(faces)}")

        for face in faces:
            x, y, w, h = map(int, face[:4])
            pose = detector.estimate_head_pose(frame, face)

            print("Head pose:", pose)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            if pose:
                label = (
                    f"Yaw: {pose['yaw']:.1f} "
                    f"Pitch: {pose['pitch']:.1f}"
                )
                cv2.putText(
                    frame, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )

        cv2.imshow(f"Head Pose - {seconds}s", frame)
        cv2.waitKey(0)

finally:
    cap.release()
    cv2.destroyAllWindows()


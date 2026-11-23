
import json
import cv2

from src.evaluation.cv_pipeline import CVPipeline
from src.evaluation.parameter_extractor import ParameterExtractor


# 1. Set the paths to our real input files.
video_path = "data/input/con.mp4"
transcript_path = "data/output/transcript_segments.json"


# 2. Load the saved speech transcript.
with open(transcript_path, "r", encoding="utf-8") as file:
    transcript_data = json.load(file)


# 3. Read the actual video duration.
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

duration_seconds = (
    frame_count / fps if fps > 0 else None
)

video.release()


# 4. Run the Computer Vision pipeline.
print("Running Computer Vision analysis...")
cv_pipeline = CVPipeline()
cv_results = cv_pipeline.analyze_video(
    video_path,
    sample_interval=2.0,
)


# 5. Extract parameters from the real results.
print("Extracting toolbox-talk parameters...")
extractor = ParameterExtractor()

results = extractor.extract_parameters(
    transcript=transcript_data,
    cv_results=cv_results,
    metadata={"duration_seconds": duration_seconds},
)


# 6. Save the combined results as JSON.
output_path = "data/output/parameter_extraction.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print("\nParameter extraction completed!")
print("Video duration:", duration_seconds, "seconds")
print("Results saved to:", output_path)
print("\nFinal results:")
print(json.dumps(results, indent=4, ensure_ascii=False))

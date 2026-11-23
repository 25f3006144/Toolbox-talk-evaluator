
from pprint import pprint

from src.cv.scene_analyzer import SceneAnalyzer


video_path = "data/input/con.mp4"

analyzer = SceneAnalyzer(
    confidence_threshold=0.50
)

result = analyzer.analyze_video(
    video_path,
    sample_interval=2.0,
)

print("\nVIDEO SCENE ANALYSIS")
print("====================")

print("\nVideo:", result["video_path"])
print(
    "Sampling interval:",
    result["sample_interval_seconds"],
    "seconds",
)

print("\nSUMMARY")
pprint(result["summary"], sort_dicts=False)

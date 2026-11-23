
import json
from pathlib import Path

import cv2

from src.video.video_processor import VideoProcessor
from src.audio.audio_processor import AudioProcessor
from src.speech.transcriber import Transcriber
from src.evaluation.cv_pipeline import CVPipeline
from src.evaluation.parameter_extractor import ParameterExtractor
from src.evaluation.scoring_engine import ScoringEngine

# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

VIDEO_PATH = Path("data/input/con.mp4")
OUTPUT_DIR = Path("data/output")

AUDIO_PATH = OUTPUT_DIR / "audio.wav"
CLEAN_AUDIO_PATH = OUTPUT_DIR / "clean_audio.wav"
ENHANCED_AUDIO_PATH = OUTPUT_DIR / "enhanced_audio.wav"

TRANSCRIPT_PATH = OUTPUT_DIR / "transcript_segments.json"
RESULT_PATH = OUTPUT_DIR / "parameter_extraction.json"
SCORING_RESULT_PATH = OUTPUT_DIR / "scoring_results.json"
SAMPLE_INTERVAL = 2.0


# --------------------------------------------------
# HELPER: GET VIDEO DURATION
# --------------------------------------------------

def get_video_duration(video_path):
    """Return video duration in seconds."""

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    video.release()

    if fps <= 0:
        raise RuntimeError(
            "Could not determine the video's frame rate."
        )

    return frame_count / fps


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    """Run the complete Toolbox Talk Evaluator backend."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validate the input video.
    print("\n[1/6] Validating input video...")

    processor = VideoProcessor(str(VIDEO_PATH))
    processor.validate()

    duration_seconds = get_video_duration(VIDEO_PATH)

    print(f"Video: {VIDEO_PATH}")
    print(f"Duration: {duration_seconds:.2f} seconds")

    # 2. Process the video's frames.
    print("\n[2/6] Processing video frames...")

    frames = processor.sample_frames(
        interval_seconds=SAMPLE_INTERVAL
    )

    if not frames:
        raise RuntimeError(
            "No frames could be read from the video."
        )

    resized_frames = processor.resize_frames(frames)
    processor.save_frames(resized_frames)

    denoised_frames = processor.reduce_noise(resized_frames)

    shot_boundaries = processor.detect_shot_boundaries(
        denoised_frames
    )

    print(f"Sampled frames: {len(resized_frames)}")
    print(f"Shot boundaries detected: {len(shot_boundaries)}")

    # 3. Extract and enhance audio.
    print("\n[3/6] Extracting and enhancing audio...")

    audio_processor = AudioProcessor(str(VIDEO_PATH))

    extracted_audio = audio_processor.extract_audio(
        output_path=str(AUDIO_PATH)
    )

    clean_audio = audio_processor.reduce_noise(
        input_path=str(extracted_audio),
        output_path=str(CLEAN_AUDIO_PATH),
    )

    enhanced_audio = audio_processor.enhance_voice(
        input_path=str(clean_audio),
        output_path=str(ENHANCED_AUDIO_PATH),
    )

    print(f"Enhanced audio saved: {enhanced_audio}")

    # 4. Convert speech to text.
    print("\n[4/6] Transcribing speech...")

    transcriber = Transcriber(model_size="small")
    transcriber.transcribe(str(enhanced_audio))

    if not TRANSCRIPT_PATH.is_file():
        raise RuntimeError(
            "Transcription finished, but the transcript JSON "
            "was not created."
        )

    with TRANSCRIPT_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        transcript_data = json.load(file)

    if not transcript_data.get("transcript", "").strip():
        raise RuntimeError(
            "The transcript is empty. Check the audio and STT output."
        )

    print("Transcript loaded successfully.")

    # 5. Run computer vision.
    print("\n[5/6] Running computer vision...")

    cv_pipeline = CVPipeline()

    cv_results = cv_pipeline.analyze_video(
        str(VIDEO_PATH),
        sample_interval=SAMPLE_INTERVAL,
    )

    print("Computer vision analysis completed.")

    # 6. Extract parameters and save the final JSON.
    print("\n[6/6] Extracting parameters and saving results...")

    extractor = ParameterExtractor()

    results = extractor.extract_parameters(
        transcript=transcript_data,
        cv_results=cv_results,
        metadata={
            "duration_seconds": duration_seconds,
        },
    )

    with RESULT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    
    # 7. Score the extracted parameters.
    print("\n[7/7] Scoring extracted parameters...")

    scoring_engine = ScoringEngine()
    scoring_results = scoring_engine.score(results)

    with SCORING_RESULT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            scoring_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("Scoring completed.")
    

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Video duration: {duration_seconds:.2f} seconds")
    print(f"Transcript: {TRANSCRIPT_PATH}")
    print(f"Final results: {RESULT_PATH}")
    print(f"Scoring results: {SCORING_RESULT_PATH}")
    print(
        "Topic detection coverage:",
        scoring_results["topic_detection_coverage_percentage"],
        "%",
    )
    print(
        "Overall quality score:",
        scoring_results["overall_score"],
    )
    print(
        "Safety topics detected:",
        sum(
            1
            for topic in results["safety_topics"].values()
            if topic["mentioned"]
        ),
        "of",
        len(results["safety_topics"]),
    )
    print("Questions detected:", results["questions"]["count"])


if __name__ == "__main__":
    main()

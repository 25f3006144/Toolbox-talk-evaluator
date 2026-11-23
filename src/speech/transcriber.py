from pathlib import Path
import json

from faster_whisper import WhisperModel


class Transcriber:
    """Converts speech audio into text."""

    def __init__(self, model_size="small"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
        
    def transcribe(self, audio_path):
        """Transcribes speech and saves timestamped results."""

        segments, info = self.model.transcribe(audio_path)

        transcript = ""
        timestamped_segments = []

        print("\nDetected language:", info.language)
        print("Language probability:", info.language_probability)

        for segment in segments:
            print(
                f"[{segment.start:.2f}s - {segment.end:.2f}s] "
                f"{segment.text}"
            )

            transcript += segment.text + " "

            timestamped_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        transcript = transcript.strip()

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "transcript.txt", "w", encoding="utf-8") as file:
            file.write(transcript)

        with open(
            output_dir / "transcript_segments.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump({
                "language": info.language,
                "language_probability": info.language_probability,
                "transcript": transcript,
                "segments": timestamped_segments
            }, file, indent=4, ensure_ascii=False)

        print("\nTranscript saved successfully!")
        print("Text: data/output/transcript.txt")
        print("Timestamps: data/output/transcript_segments.json")

        return transcript
    
if __name__ == "__main__":
    transcriber = Transcriber()

    audio_path = "data/output/enhanced_audio.wav"

    text = transcriber.transcribe(audio_path)

    print("\nFinal Transcript:")
    print(text)
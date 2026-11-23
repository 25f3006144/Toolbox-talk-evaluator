from pathlib import Path
import subprocess

class AudioProcessor:
    """Handles audio processing for toolbox talk videos."""

    def __init__(self, video_path):
        self.video_path = Path(video_path)

    def has_audio(self):
        """Check whether the video contains an audio stream."""

        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        return result.stdout.strip() == "audio"
    def extract_audio(self, output_path="data/output/audio.wav"):
        """Extract and standardize audio using FFmpeg."""

        if not self.has_audio():
            raise ValueError("Video does not contain an audio stream.")

        command = [
            "ffmpeg",
            "-y",
            "-i", str(self.video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path
        ]

        subprocess.run(command, check=True)

        return Path(output_path)
    
    def reduce_noise(
        self,
        input_path="data/output/audio.wav",
        output_path="data/output/clean_audio.wav"
    ):
        """Reduce background noise from the extracted audio."""

        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", "afftdn=nr=20:nf=-35:tn=1",
            output_path
        ]

        subprocess.run(command, check=True)

        return Path(output_path)  
      
    def enhance_voice(
        self,
        input_path="data/output/clean_audio.wav",
        output_path="data/output/enhanced_audio.wav"
    ):
        """Enhance speech clarity in the audio."""

        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", "highpass=f=80,lowpass=f=12000",
            output_path
        ]

        subprocess.run(command, check=True)

        return Path(output_path)
    
    
   
    
    
    
    

     
    
     
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.video.video_processor import VideoProcessor


class TestVideoProcessor(unittest.TestCase):

    def test_supported_video_file(self):
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "test.mp4"
            video_path.touch()

            processor = VideoProcessor(video_path)

            self.assertTrue(processor.validate())
    
    def test_resize_frames_preserves_aspect_ratio(self):
        import numpy as np

        processor = VideoProcessor("dummy.mp4")

        # Create a sample frame: width 864, height 480
        frame = np.zeros((480, 864, 3), dtype=np.uint8)

        resized_frames = processor.resize_frames([frame])

        resized_frame = resized_frames[0]
        new_height, new_width = resized_frame.shape[:2]

        self.assertLessEqual(new_width, 1920)
        self.assertLessEqual(new_height, 1080)

        original_ratio = 864 / 480
        resized_ratio = new_width / new_height

        self.assertAlmostEqual(
            original_ratio,
            resized_ratio,
            places=2
        )
        


if __name__ == "__main__":
    unittest.main()


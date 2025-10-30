"""Tests for AudioProcessor component.

Following TDD methodology:
1. RED: Write failing tests
2. GREEN: Make tests pass
3. REFACTOR: Clean up code
"""

import pytest
from pathlib import Path
from moviepy import VideoFileClip
from src.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""

    def test_strip_audio_removes_audio_track(self, sample_video_with_audio, output_path):
        """Test that strip_audio() removes the audio track from video."""
        processor = AudioProcessor()

        result_path = processor.strip_audio(str(sample_video_with_audio), str(output_path))

        # Verify output file exists
        assert Path(result_path).exists()

        # Verify audio track is removed
        video = VideoFileClip(result_path)
        assert video.audio is None
        video.close()

    def test_strip_audio_preserves_video_properties(self, sample_video_with_audio, temp_dir):
        """Test that strip_audio() preserves video FPS, resolution, and codec."""
        processor = AudioProcessor()
        output_path = temp_dir / "output_stripped.mp4"

        # Get original video properties
        original_video = VideoFileClip(str(sample_video_with_audio))
        original_fps = original_video.fps
        original_size = original_video.size
        original_duration = original_video.duration
        original_video.close()

        # Strip audio
        result_path = processor.strip_audio(str(sample_video_with_audio), str(output_path))

        # Verify properties preserved
        result_video = VideoFileClip(result_path)
        assert result_video.fps == original_fps
        assert result_video.size == original_size
        assert abs(result_video.duration - original_duration) < 0.1  # ±0.1s tolerance
        result_video.close()

    def test_strip_audio_returns_valid_path(self, sample_video_with_audio, output_path):
        """Test that strip_audio() returns a valid Path object."""
        processor = AudioProcessor()

        result = processor.strip_audio(str(sample_video_with_audio), str(output_path))

        assert isinstance(result, str)
        assert Path(result).exists()
        assert Path(result).is_file()

    def test_overlay_voiceover_adds_audio(self, sample_video_silent, sample_audio, output_path):
        """Test that overlay_voiceover() adds audio track to silent video."""
        processor = AudioProcessor()

        result_path = processor.overlay_voiceover(
            str(sample_video_silent),
            str(sample_audio),
            str(output_path)
        )

        # Verify output file exists
        assert Path(result_path).exists()

        # Verify audio track exists
        video = VideoFileClip(result_path)
        assert video.audio is not None
        video.close()

    def test_overlay_voiceover_matches_duration(self, sample_video_silent, sample_audio, output_path):
        """Test that overlay_voiceover() creates video with matching audio duration."""
        processor = AudioProcessor()

        # Get original durations
        video = VideoFileClip(str(sample_video_silent))
        video_duration = video.duration
        video.close()

        # Overlay voiceover
        result_path = processor.overlay_voiceover(
            str(sample_video_silent),
            str(sample_audio),
            str(output_path)
        )

        # Verify durations match
        result_video = VideoFileClip(result_path)
        assert result_video.audio is not None
        assert abs(result_video.duration - video_duration) < 0.1  # ±0.1s tolerance
        result_video.close()

    def test_overlay_voiceover_preserves_video_quality(self, sample_video_silent, sample_audio, output_path):
        """Test that overlay_voiceover() preserves video properties."""
        processor = AudioProcessor()

        # Get original video properties
        original_video = VideoFileClip(str(sample_video_silent))
        original_fps = original_video.fps
        original_size = original_video.size
        original_video.close()

        # Overlay voiceover
        result_path = processor.overlay_voiceover(
            str(sample_video_silent),
            str(sample_audio),
            str(output_path)
        )

        # Verify properties preserved
        result_video = VideoFileClip(result_path)
        assert result_video.fps == original_fps
        assert result_video.size == original_size
        result_video.close()

    def test_handles_missing_video_file(self, sample_audio, output_path):
        """Test that methods raise FileNotFoundError for missing video file."""
        processor = AudioProcessor()
        nonexistent_video = "/tmp/nonexistent_video.mp4"

        with pytest.raises(FileNotFoundError):
            processor.strip_audio(nonexistent_video, str(output_path))

        with pytest.raises(FileNotFoundError):
            processor.overlay_voiceover(nonexistent_video, str(sample_audio), str(output_path))

    def test_handles_missing_audio_file(self, sample_video_silent, output_path):
        """Test that overlay_voiceover() raises FileNotFoundError for missing audio."""
        processor = AudioProcessor()
        nonexistent_audio = "/tmp/nonexistent_audio.mp3"

        with pytest.raises(FileNotFoundError):
            processor.overlay_voiceover(str(sample_video_silent), nonexistent_audio, str(output_path))

    def test_handles_corrupted_video_file(self, corrupted_video, sample_audio, output_path, temp_dir):
        """Test that methods handle corrupted video files gracefully."""
        processor = AudioProcessor()

        # Should raise an exception (ValueError or IOError) when processing corrupted file
        with pytest.raises((ValueError, IOError, RuntimeError)):
            processor.strip_audio(str(corrupted_video), str(temp_dir / "output1.mp4"))

        with pytest.raises((ValueError, IOError, RuntimeError)):
            processor.overlay_voiceover(str(corrupted_video), str(sample_audio), str(temp_dir / "output2.mp4"))

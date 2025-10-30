"""Pytest configuration and fixtures for sora-extend tests."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from moviepy import VideoClip, AudioClip, ColorClip


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_video_with_audio(temp_dir):
    """Generate a 2-second test video with audio."""
    video_path = temp_dir / "video_with_audio.mp4"

    # Create a simple 2-second video (640x480, red color)
    def make_frame(t):
        return np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)

    video = VideoClip(make_frame, duration=2)
    video = video.with_fps(24)

    # Create a simple audio clip (440 Hz tone)
    def make_audio_frame(t):
        frequency = 440  # A4 note
        return np.sin(2 * np.pi * frequency * t)

    audio = AudioClip(make_audio_frame, duration=2, fps=44100)
    video = video.with_audio(audio)

    # Write video file
    video.write_videofile(
        str(video_path),
        codec="libx264",
        audio_codec="aac",
        logger=None  # Suppress MoviePy output
    )

    video.close()

    return video_path


@pytest.fixture
def sample_video_silent(temp_dir):
    """Generate a 2-second test video without audio."""
    video_path = temp_dir / "video_silent.mp4"

    # Create a simple 2-second video (640x480, blue color)
    clip = ColorClip(size=(640, 480), color=(0, 0, 255), duration=2)
    clip = clip.with_fps(24)

    # Write video file without audio
    clip.write_videofile(
        str(video_path),
        codec="libx264",
        logger=None
    )

    clip.close()

    return video_path


@pytest.fixture
def sample_audio(temp_dir):
    """Generate a 2-second audio file."""
    audio_path = temp_dir / "sample_audio.mp3"

    # Create a simple 2-second audio clip (880 Hz tone)
    def make_audio_frame(t):
        frequency = 880  # A5 note
        return np.sin(2 * np.pi * frequency * t)

    audio = AudioClip(make_audio_frame, duration=2, fps=44100)

    # Write audio file
    audio.write_audiofile(str(audio_path), logger=None)
    audio.close()

    return audio_path


@pytest.fixture
def corrupted_video(temp_dir):
    """Create a corrupted video file for error testing."""
    video_path = temp_dir / "corrupted.mp4"

    # Write invalid data
    with open(video_path, 'wb') as f:
        f.write(b'not a valid video file')

    return video_path


@pytest.fixture
def output_path(temp_dir):
    """Provide a path for output files."""
    return temp_dir / "output.mp4"

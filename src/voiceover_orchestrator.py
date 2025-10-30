"""VoiceoverOrchestrator component for coordinating voiceover workflow.

This module orchestrates the complete workflow of generating voiceover
and applying it to video files by coordinating AudioProcessor and
VoiceoverGenerator components.
"""

import tempfile
from pathlib import Path
from typing import Union
from audio_processor import AudioProcessor
from voiceover_generator import VoiceoverGenerator


class VoiceoverOrchestrator:
    """Orchestrates the complete voiceover generation and application workflow."""

    def __init__(self, api_key: str, voice_id: str):
        """Initialize VoiceoverOrchestrator with API credentials.

        Args:
            api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID to use for voiceover
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.audio_processor = AudioProcessor()
        self.voiceover_generator = VoiceoverGenerator(api_key, voice_id)

    def process(
        self,
        video_path: Union[str, Path],
        text: str,
        output_path: Union[str, Path]
    ) -> str:
        """Generate voiceover and apply it to video.

        This method orchestrates the complete workflow:
        1. Validates inputs
        2. Generates voiceover audio from text
        3. Strips existing audio from video
        4. Overlays new voiceover onto video
        5. Cleans up temporary files

        Args:
            video_path: Path to input video file
            text: Text to convert to voiceover
            output_path: Path where final video will be saved

        Returns:
            str: Path to the output video file

        Raises:
            FileNotFoundError: If video_path does not exist
            ValueError: If text is empty
            RuntimeError: If voiceover generation or video processing fails
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # Validate inputs
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp())
        temp_audio_path = temp_dir / "voiceover.mp3"
        temp_video_path = temp_dir / "video_silent.mp4"

        try:
            # Step 1: Generate voiceover audio from text
            try:
                self.voiceover_generator.generate(
                    text=text,
                    output_path=str(temp_audio_path)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate voiceover: {e}")

            # Step 2: Strip existing audio from video
            try:
                self.audio_processor.strip_audio(
                    video_path=str(video_path),
                    output_path=str(temp_video_path)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to process video: {e}")

            # Step 3: Overlay new voiceover onto silent video
            try:
                result_path = self.audio_processor.overlay_voiceover(
                    video_path=str(temp_video_path),
                    audio_path=str(temp_audio_path),
                    output_path=str(output_path)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to overlay voiceover: {e}")

            return result_path

        finally:
            # Clean up temporary files
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            if temp_video_path.exists():
                temp_video_path.unlink()
            # Remove temp directory if it's empty
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty or already removed

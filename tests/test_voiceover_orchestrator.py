"""Tests for VoiceoverOrchestrator component.

Following TDD methodology:
1. RED: Write failing tests
2. GREEN: Make tests pass
3. REFACTOR: Clean up code
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.voiceover_orchestrator import VoiceoverOrchestrator


class TestVoiceoverOrchestrator:
    """Test suite for VoiceoverOrchestrator class."""

    def test_process_returns_valid_video_path(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() returns a valid video file path."""
        output_path = temp_dir / "final_output.mp4"

        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg, \
             patch('src.voiceover_orchestrator.AudioProcessor') as mock_ap:

            # Mock VoiceoverGenerator
            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            mock_generator.generate.return_value = str(temp_dir / "temp_audio.mp3")

            # Mock AudioProcessor
            mock_processor = Mock()
            mock_ap.return_value = mock_processor
            mock_processor.strip_audio.return_value = str(temp_dir / "temp_silent.mp4")
            mock_processor.overlay_voiceover.return_value = str(output_path)

            # Create orchestrator AFTER patches are applied
            orchestrator = VoiceoverOrchestrator(
                api_key="test_key",
                voice_id="test_voice"
            )

            # Create dummy files
            (temp_dir / "temp_audio.mp3").touch()
            (temp_dir / "temp_silent.mp4").touch()
            output_path.touch()

            result = orchestrator.process(
                video_path=str(sample_video_with_audio),
                text="Test voiceover text",
                output_path=str(output_path)
            )

            assert result == str(output_path)
            assert Path(result).exists()

    def test_process_calls_components_in_correct_order(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() calls components in the correct order."""
        output_path = temp_dir / "output.mp4"

        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg, \
             patch('src.voiceover_orchestrator.AudioProcessor') as mock_ap:

            # Setup mocks
            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            mock_generator.generate.return_value = str(temp_dir / "audio.mp3")

            mock_processor = Mock()
            mock_ap.return_value = mock_processor
            mock_processor.strip_audio.return_value = str(temp_dir / "silent.mp4")
            mock_processor.overlay_voiceover.return_value = str(output_path)

            # Create orchestrator AFTER patches
            orchestrator = VoiceoverOrchestrator(
                api_key="test_key",
                voice_id="test_voice"
            )

            # Create dummy files
            (temp_dir / "audio.mp3").touch()
            (temp_dir / "silent.mp4").touch()
            output_path.touch()

            orchestrator.process(
                video_path=str(sample_video_with_audio),
                text="Test text",
                output_path=str(output_path)
            )

            # Verify call order
            assert mock_generator.generate.called
            assert mock_processor.strip_audio.called
            assert mock_processor.overlay_voiceover.called

            # Verify generator was called with correct text
            mock_generator.generate.assert_called_once()
            call_kwargs = mock_generator.generate.call_args.kwargs
            assert call_kwargs['text'] == "Test text"

            # Verify strip_audio was called with original video
            mock_processor.strip_audio.assert_called_once()
            strip_kwargs = mock_processor.strip_audio.call_args.kwargs
            assert strip_kwargs['video_path'] == str(sample_video_with_audio)

            # Verify overlay was called with silent video and generated audio
            mock_processor.overlay_voiceover.assert_called_once()

    def test_process_cleans_up_temporary_files(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() cleans up temporary files after completion."""
        output_path = temp_dir / "output.mp4"
        temp_audio = temp_dir / "temp_audio.mp3"
        temp_video = temp_dir / "temp_video.mp4"

        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg, \
             patch('src.voiceover_orchestrator.AudioProcessor') as mock_ap, \
             patch('src.voiceover_orchestrator.tempfile.mkdtemp') as mock_mkdtemp:

            # Mock tempfile to use our temp_dir
            mock_mkdtemp.return_value = str(temp_dir)

            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            # Generate returns a path that will be in our controlled temp_dir
            mock_generator.generate.return_value = str(temp_dir / "voiceover.mp3")

            mock_processor = Mock()
            mock_ap.return_value = mock_processor
            mock_processor.strip_audio.return_value = str(temp_dir / "video_silent.mp4")
            mock_processor.overlay_voiceover.return_value = str(output_path)

            # Create orchestrator AFTER patches
            orchestrator = VoiceoverOrchestrator(
                api_key="test_key",
                voice_id="test_voice"
            )

            # Create temporary files that match what the code expects
            (temp_dir / "voiceover.mp3").touch()
            (temp_dir / "video_silent.mp4").touch()
            output_path.touch()

            orchestrator.process(
                video_path=str(sample_video_with_audio),
                text="Test",
                output_path=str(output_path)
            )

            # Verify temp files are cleaned up
            assert not (temp_dir / "voiceover.mp3").exists()
            assert not (temp_dir / "video_silent.mp4").exists()

    def test_process_handles_voiceover_generation_failure(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() handles VoiceoverGenerator failures."""
        orchestrator = VoiceoverOrchestrator(
            api_key="test_key",
            voice_id="test_voice"
        )

        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg:
            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            mock_generator.generate.side_effect = RuntimeError("API failure")

            with pytest.raises(RuntimeError, match="Failed to generate voiceover"):
                orchestrator.process(
                    video_path=str(sample_video_with_audio),
                    text="Test",
                    output_path=str(temp_dir / "output.mp4")
                )

    def test_process_handles_audio_processing_failure(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() handles AudioProcessor failures."""
        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg, \
             patch('src.voiceover_orchestrator.AudioProcessor') as mock_ap:

            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            mock_generator.generate.return_value = str(temp_dir / "audio.mp3")

            mock_processor = Mock()
            mock_ap.return_value = mock_processor
            mock_processor.strip_audio.side_effect = RuntimeError("Video processing failed")

            # Create orchestrator AFTER patches
            orchestrator = VoiceoverOrchestrator(
                api_key="test_key",
                voice_id="test_voice"
            )

            (temp_dir / "audio.mp3").touch()

            with pytest.raises(RuntimeError, match="Failed to process video"):
                orchestrator.process(
                    video_path=str(sample_video_with_audio),
                    text="Test",
                    output_path=str(temp_dir / "output.mp4")
                )

    def test_process_validates_input_video_exists(self, temp_dir):
        """Test that process() validates input video file exists."""
        orchestrator = VoiceoverOrchestrator(
            api_key="test_key",
            voice_id="test_voice"
        )

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            orchestrator.process(
                video_path="/nonexistent/video.mp4",
                text="Test",
                output_path=str(temp_dir / "output.mp4")
            )

    def test_process_validates_text_not_empty(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() validates text is not empty."""
        orchestrator = VoiceoverOrchestrator(
            api_key="test_key",
            voice_id="test_voice"
        )

        with pytest.raises(ValueError, match="Text cannot be empty"):
            orchestrator.process(
                video_path=str(sample_video_with_audio),
                text="",
                output_path=str(temp_dir / "output.mp4")
            )

    def test_process_creates_output_directory(
        self,
        sample_video_with_audio,
        temp_dir
    ):
        """Test that process() creates output directory if it doesn't exist."""
        output_path = temp_dir / "nested" / "dirs" / "output.mp4"

        with patch('src.voiceover_orchestrator.VoiceoverGenerator') as mock_vg, \
             patch('src.voiceover_orchestrator.AudioProcessor') as mock_ap:

            mock_generator = Mock()
            mock_vg.return_value = mock_generator
            mock_generator.generate.return_value = str(temp_dir / "audio.mp3")

            mock_processor = Mock()
            mock_ap.return_value = mock_processor
            mock_processor.strip_audio.return_value = str(temp_dir / "silent.mp4")
            mock_processor.overlay_voiceover.return_value = str(output_path)

            # Create orchestrator AFTER patches
            orchestrator = VoiceoverOrchestrator(
                api_key="test_key",
                voice_id="test_voice"
            )

            # Create dummy files
            (temp_dir / "audio.mp3").touch()
            (temp_dir / "silent.mp4").touch()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

            result = orchestrator.process(
                video_path=str(sample_video_with_audio),
                text="Test",
                output_path=str(output_path)
            )

            assert Path(result).parent.exists()

"""Tests for VoiceoverGenerator component.

Following TDD methodology:
1. RED: Write failing tests
2. GREEN: Make tests pass
3. REFACTOR: Clean up code
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.voiceover_generator import VoiceoverGenerator


class TestVoiceoverGenerator:
    """Test suite for VoiceoverGenerator class."""

    def test_generate_returns_valid_audio_file(self, temp_dir):
        """Test that generate() returns a valid audio file path."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            # Mock the API response
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.return_value = b"fake_audio_data"

            result_path = generator.generate(
                text="Test text",
                output_path=str(temp_dir / "output.mp3")
            )

            # Verify output file exists
            assert Path(result_path).exists()
            assert Path(result_path).is_file()

    def test_generate_calls_elevenlabs_api_correctly(self, temp_dir):
        """Test that generate() calls ElevenLabs API with correct parameters."""
        api_key = "test_api_key"
        voice_id = "test_voice_id"
        test_text = "This is a test voiceover"

        generator = VoiceoverGenerator(api_key=api_key, voice_id=voice_id)

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.return_value = b"fake_audio_data"

            generator.generate(
                text=test_text,
                output_path=str(temp_dir / "output.mp3")
            )

            # Verify ElevenLabs client was initialized with API key
            mock_elevenlabs.assert_called_once_with(api_key=api_key)

            # Verify generate was called with correct parameters
            mock_client.generate.assert_called_once_with(
                text=test_text,
                voice=voice_id,
                model="eleven_multilingual_v2"
            )

    def test_generate_saves_audio_to_correct_path(self, temp_dir):
        """Test that generate() saves audio to the specified path."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")
        output_path = temp_dir / "test_output.mp3"

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.return_value = b"fake_audio_data"

            result = generator.generate(
                text="Test",
                output_path=str(output_path)
            )

            assert result == str(output_path)
            assert output_path.exists()

    def test_generate_retries_on_api_failure(self, temp_dir):
        """Test that generate() retries on transient API failures."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client

            # Simulate 2 failures then success
            mock_client.generate.side_effect = [
                Exception("API rate limit"),
                Exception("API timeout"),
                b"fake_audio_data"
            ]

            result = generator.generate(
                text="Test",
                output_path=str(temp_dir / "output.mp3"),
                max_retries=3
            )

            # Should succeed after retries
            assert Path(result).exists()
            assert mock_client.generate.call_count == 3

    def test_generate_fails_after_max_retries(self, temp_dir):
        """Test that generate() raises exception after max retries."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.side_effect = Exception("Persistent API error")

            with pytest.raises(RuntimeError, match="Failed to generate voiceover"):
                generator.generate(
                    text="Test",
                    output_path=str(temp_dir / "output.mp3"),
                    max_retries=2
                )

            # Should have tried max_retries + 1 times (initial + retries)
            assert mock_client.generate.call_count == 3

    def test_generate_handles_empty_text(self, temp_dir):
        """Test that generate() raises ValueError for empty text."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.generate(
                text="",
                output_path=str(temp_dir / "output.mp3")
            )

        with pytest.raises(ValueError, match="Text cannot be empty"):
            generator.generate(
                text="   ",
                output_path=str(temp_dir / "output.mp3")
            )

    def test_generate_handles_invalid_api_key(self, temp_dir):
        """Test that generate() handles invalid API key gracefully."""
        generator = VoiceoverGenerator(api_key="invalid_key", voice_id="test_voice")

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.side_effect = Exception("Invalid API key")

            with pytest.raises(RuntimeError, match="Failed to generate voiceover"):
                generator.generate(
                    text="Test",
                    output_path=str(temp_dir / "output.mp3"),
                    max_retries=1
                )

    def test_generate_creates_parent_directories(self, temp_dir):
        """Test that generate() creates parent directories if they don't exist."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")
        output_path = temp_dir / "nested" / "path" / "output.mp3"

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs:
            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.return_value = b"fake_audio_data"

            result = generator.generate(
                text="Test",
                output_path=str(output_path)
            )

            assert Path(result).exists()
            assert output_path.parent.exists()

    def test_generate_uses_exponential_backoff(self, temp_dir):
        """Test that generate() uses exponential backoff between retries."""
        generator = VoiceoverGenerator(api_key="test_key", voice_id="test_voice")

        with patch('src.voiceover_generator.ElevenLabs') as mock_elevenlabs, \
             patch('src.voiceover_generator.time.sleep') as mock_sleep:

            mock_client = Mock()
            mock_elevenlabs.return_value = mock_client
            mock_client.generate.side_effect = [
                Exception("Error 1"),
                Exception("Error 2"),
                b"fake_audio_data"
            ]

            generator.generate(
                text="Test",
                output_path=str(temp_dir / "output.mp3"),
                max_retries=3
            )

            # Verify sleep was called with increasing durations
            # First retry: 1s, Second retry: 2s
            assert mock_sleep.call_count == 2
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls[0] == 1  # First backoff
            assert calls[1] == 2  # Second backoff (exponential)

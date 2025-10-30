"""VoiceoverGenerator component for text-to-speech using ElevenLabs.

This module provides functionality to generate voiceover audio from text
using the ElevenLabs API with retry logic and error handling.
"""

import time
from pathlib import Path
from typing import Union
from elevenlabs import ElevenLabs


class VoiceoverGenerator:
    """Generates voiceover audio from text using ElevenLabs API."""

    def __init__(self, api_key: str, voice_id: str):
        """Initialize VoiceoverGenerator with API credentials.

        Args:
            api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID to use for generation
        """
        self.api_key = api_key
        self.voice_id = voice_id

    def generate(
        self,
        text: str,
        output_path: Union[str, Path],
        max_retries: int = 3
    ) -> str:
        """Generate voiceover audio from text.

        Args:
            text: Text to convert to speech
            output_path: Path where audio file will be saved
            max_retries: Maximum number of retry attempts on failure

        Returns:
            str: Path to the generated audio file

        Raises:
            ValueError: If text is empty or whitespace-only
            RuntimeError: If generation fails after max retries
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        output_path = Path(output_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Initialize ElevenLabs client
                client = ElevenLabs(api_key=self.api_key)

                # Generate audio using text_to_speech API
                audio_data = client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2"
                )

                # Save audio to file
                with open(output_path, 'wb') as f:
                    # Handle both bytes and iterator responses
                    if isinstance(audio_data, bytes):
                        f.write(audio_data)
                    else:
                        # audio_data is an iterator of audio chunks
                        for chunk in audio_data:
                            if chunk:
                                f.write(chunk)

                return str(output_path)

            except Exception as e:
                last_exception = e

                # If this was the last attempt, raise error
                if attempt == max_retries:
                    break

                # Exponential backoff: 1s, 2s, 4s, 8s, etc.
                backoff_time = 2 ** attempt
                time.sleep(backoff_time)

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed to generate voiceover after {max_retries + 1} attempts: {last_exception}"
        )

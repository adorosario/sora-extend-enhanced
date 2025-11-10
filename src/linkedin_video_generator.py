"""LinkedIn Video Generator - End-to-end pipeline for creating videos from LinkedIn posts.

This module orchestrates the complete workflow:
1. Generate script from LinkedIn post (GPT)
2. Generate studio B-roll (Sora) + voiceover (ElevenLabs) in parallel
3. Composite video + audio
"""

import os
import sys
import time
import asyncio
import argparse
import requests
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from moviepy import VideoFileClip, AudioFileClip

from script_generator import ScriptGenerator
from voiceover_generator import VoiceoverGenerator


class LinkedInVideoGenerator:
    """End-to-end generator for creating videos from LinkedIn posts."""

    def __init__(
        self,
        openai_api_key: str,
        elevenlabs_api_key: str,
        elevenlabs_voice_id: str,
        output_dir: str = "./output/linkedin_videos"
    ):
        """Initialize the video generator.

        Args:
            openai_api_key: OpenAI API key for GPT and Sora
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_voice_id: ElevenLabs voice ID
            output_dir: Directory for output files
        """
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)

        self.script_generator = ScriptGenerator(api_key=openai_api_key)
        self.voiceover_generator = VoiceoverGenerator(
            api_key=elevenlabs_api_key,
            voice_id=elevenlabs_voice_id
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_video(
        self,
        linkedin_post_path: str,
        video_name: Optional[str] = None,
        target_duration: int = 60,
        keep_intermediates: bool = True
    ) -> str:
        """Generate complete video from LinkedIn post.

        Args:
            linkedin_post_path: Path to .txt file with LinkedIn post
            video_name: Name for output video (default: auto-generated from timestamp)
            target_duration: Target video duration in seconds
            keep_intermediates: Keep intermediate files (script, B-roll, voiceover)

        Returns:
            str: Path to final video file

        Raises:
            FileNotFoundError: If linkedin_post_path doesn't exist
            RuntimeError: If video generation fails
        """
        start_time = time.time()

        # Read LinkedIn post
        linkedin_post_path = Path(linkedin_post_path)
        if not linkedin_post_path.exists():
            raise FileNotFoundError(f"LinkedIn post not found: {linkedin_post_path}")

        with open(linkedin_post_path, 'r') as f:
            linkedin_post = f.read()

        # Create output subdirectory for this video
        if video_name is None:
            video_name = f"video_{int(time.time())}"

        video_dir = self.output_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"LinkedIn Video Generator")
        print(f"{'='*60}")
        print(f"Input: {linkedin_post_path}")
        print(f"Output Directory: {video_dir}")
        print(f"Target Duration: {target_duration}s\n")

        # STEP 1: Generate script
        print("STEP 1/4: Generating script from LinkedIn post...")
        try:
            script_data = self.script_generator.generate_script(
                linkedin_post=linkedin_post,
                target_duration=target_duration
            )

            print(f"  ✓ Script generated")
            print(f"    Theme: {script_data['visual_theme']}")
            print(f"    Words: {script_data['word_count']}")
            print(f"    Duration: ~{script_data['estimated_duration']}s")

            # Save script
            script_path = video_dir / "script.json"
            self.script_generator.save_script(script_data, script_path)
            print(f"    Saved: {script_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to generate script: {e}")

        # STEP 2: Parallel generation (Sora B-roll + ElevenLabs voiceover)
        print("\nSTEP 2/4: Generating studio B-roll and voiceover in parallel...")

        studio_prompt = self.script_generator.get_studio_prompt(
            script_data['visual_theme'],
            script_data['topic']
        )
        broll_path = video_dir / "studio_broll.mp4"
        voiceover_path = video_dir / "voiceover.mp3"

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_broll = executor.submit(
                    self._generate_sora_video,
                    studio_prompt,
                    target_duration,
                    broll_path
                )
                future_voiceover = executor.submit(
                    self._generate_voiceover,
                    script_data['script'],
                    voiceover_path
                )

                # Wait for both to complete
                for future in as_completed([future_broll, future_voiceover]):
                    result = future.result()  # Will raise exception if task failed

            print(f"  ✓ Studio B-roll generated: {broll_path}")
            print(f"  ✓ Voiceover generated: {voiceover_path}")

        except Exception as e:
            raise RuntimeError(f"Failed during parallel generation: {e}")

        # STEP 3: Composite video + audio
        print("\nSTEP 3/4: Compositing video and audio...")

        final_path = video_dir / f"{video_name}.mp4"

        try:
            self._composite_video(broll_path, voiceover_path, final_path, target_duration)
            print(f"  ✓ Video composited: {final_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to composite video: {e}")

        # STEP 4: Cleanup (optional)
        if not keep_intermediates:
            print("\nSTEP 4/4: Cleaning up intermediate files...")
            broll_path.unlink(missing_ok=True)
            voiceover_path.unlink(missing_ok=True)
            print("  ✓ Intermediate files removed")
        else:
            print("\nSTEP 4/4: Keeping intermediate files")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Video generation complete!")
        print(f"  Time: {elapsed:.1f}s (~{elapsed/60:.1f} min)")
        print(f"  Output: {final_path}")
        print(f"  Size: {final_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"{'='*60}\n")

        return str(final_path)

    def _generate_sora_video(self, prompt: str, duration: int, output_path: Path) -> str:
        """Generate video using Sora API (via raw HTTP requests).

        Args:
            prompt: Sora prompt for video generation
            duration: Video duration in seconds
            output_path: Where to save the video

        Returns:
            str: Path to generated video

        Raises:
            RuntimeError: If video generation fails
        """
        print(f"    [Sora] Generating {duration}s studio B-roll...")

        try:
            # Setup API request
            api_base = "https://api.openai.com/v1"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
            }

            # Create Sora video job via POST /videos
            files = {
                "model": (None, "sora-2"),
                "prompt": (None, prompt),
                "seconds": (None, str(duration)),
                "size": (None, "1280x720"),  # Landscape HD (standard YouTube/social)
            }

            create_url = f"{api_base}/videos"
            response = requests.post(create_url, headers=headers, files=files, timeout=30)

            if response.status_code >= 400:
                raise RuntimeError(f"Create video failed: {response.text}")

            job = response.json()
            job_id = job["id"]
            print(f"    [Sora] Job created: {job_id}")

            # Poll for completion
            max_wait = 30 * 60  # 30 minutes max (Sora can be slow during peak times)
            start_time = time.time()
            poll_interval = 15  # Check every 15 seconds

            while True:
                if time.time() - start_time > max_wait:
                    raise RuntimeError("Sora job timed out after 30 minutes")

                # Check job status via GET /videos/{id}
                status_url = f"{api_base}/videos/{job_id}"
                status_response = requests.get(status_url, headers=headers, timeout=30)

                if status_response.status_code >= 400:
                    raise RuntimeError(f"Retrieve video failed: {status_response.text}")

                video_status = status_response.json()

                if video_status["status"] == "completed":
                    # Download video via GET /videos/{id}/content
                    download_url = f"{api_base}/videos/{job_id}/content"
                    params = {"variant": "video"}

                    with requests.get(download_url, headers=headers, params=params, stream=True, timeout=300) as video_response:
                        if video_response.status_code >= 400:
                            raise RuntimeError(f"Download failed: {video_response.text}")

                        with open(output_path, 'wb') as f:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                    print(f"    [Sora] ✓ Video downloaded")
                    return str(output_path)

                elif video_status["status"] == "failed":
                    error_msg = video_status.get("error", {}).get("message", "Unknown error")
                    raise RuntimeError(f"Sora job failed: {error_msg}")

                # Still processing
                elapsed = int(time.time() - start_time)
                print(f"    [Sora] Status: {video_status['status']} ({elapsed}s elapsed)")
                time.sleep(poll_interval)

        except Exception as e:
            raise RuntimeError(f"Sora video generation failed: {e}")

    def _generate_voiceover(self, script: str, output_path: Path) -> str:
        """Generate voiceover using ElevenLabs.

        Args:
            script: Script text to convert to speech
            output_path: Where to save the audio

        Returns:
            str: Path to generated audio

        Raises:
            RuntimeError: If voiceover generation fails
        """
        print(f"    [ElevenLabs] Generating voiceover...")

        try:
            result = self.voiceover_generator.generate(
                text=script,
                output_path=output_path
            )
            print(f"    [ElevenLabs] ✓ Voiceover generated")
            return result

        except Exception as e:
            raise RuntimeError(f"Voiceover generation failed: {e}")

    def _composite_video(self, video_path: Path, audio_path: Path, output_path: Path, target_duration: int) -> str:
        """Composite video and audio using MoviePy.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Where to save final video
            target_duration: Target duration in seconds (will trim to this length)

        Returns:
            str: Path to final video

        Raises:
            RuntimeError: If compositing fails
        """
        try:
            # Load video and strip any existing audio
            video = VideoFileClip(str(video_path))
            if video.audio is not None:
                video = video.without_audio()

            # Load audio
            audio = AudioFileClip(str(audio_path))

            # Enforce target duration by trimming both video and audio
            # This ensures the final video is exactly target_duration seconds
            if audio.duration > target_duration:
                audio = audio.subclipped(0, target_duration)

            if video.duration > target_duration:
                video = video.subclipped(0, target_duration)

            # If audio is shorter than video, trim video to match audio
            if video.duration > audio.duration:
                video = video.subclipped(0, audio.duration)

            # Set audio (MoviePy 2.x uses with_audio)
            final = video.with_audio(audio)

            # Write output
            final.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=24,
                preset='medium',
                logger=None  # Suppress MoviePy progress bars
            )

            # Clean up
            video.close()
            audio.close()
            final.close()

            return str(output_path)

        except Exception as e:
            raise RuntimeError(f"Video compositing failed: {e}")


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate professional videos from LinkedIn posts"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to LinkedIn post .txt file"
    )
    parser.add_argument(
        "--name",
        help="Name for output video (default: auto-generated)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Target video duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output/linkedin_videos",
        help="Output directory (default: ./output/linkedin_videos)"
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files (script, B-roll, voiceover)"
    )

    args = parser.parse_args()

    # Load environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    # Validate environment
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if not elevenlabs_api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        sys.exit(1)
    if not elevenlabs_voice_id:
        print("Error: ELEVENLABS_VOICE_ID environment variable not set")
        sys.exit(1)

    # Create generator
    generator = LinkedInVideoGenerator(
        openai_api_key=openai_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        output_dir=args.output_dir
    )

    # Generate video
    try:
        final_video = generator.generate_video(
            linkedin_post_path=args.input,
            video_name=args.name,
            target_duration=args.duration,
            keep_intermediates=args.keep_intermediates
        )

        print(f"\n🎬 Success! Video ready at: {final_video}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

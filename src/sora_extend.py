#!/usr/bin/env python3
"""
Sora 2 AI-Planned Video Extension
Extends OpenAI's Sora 2 video generation beyond 12 seconds by chaining multiple segments
with intelligent planning and continuity.
"""

import sys
from pathlib import Path

# Add src directory to path if running as a script
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import os
import re
import json
import time
import mimetypes
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from functools import wraps

import requests
import cv2
from moviepy import VideoFileClip, concatenate_videoclips
from openai import OpenAI
from dotenv import load_dotenv

# Voiceover components (optional)
try:
    # Try relative import first (when running as module)
    from voiceover_orchestrator import VoiceoverOrchestrator
    VOICEOVER_AVAILABLE = True
except ImportError:
    try:
        # Fall back to absolute import (when running from parent directory)
        from src.voiceover_orchestrator import VoiceoverOrchestrator
        VOICEOVER_AVAILABLE = True
    except ImportError:
        VOICEOVER_AVAILABLE = False


def retry_with_backoff(max_retries=5, initial_delay=2, backoff_factor=2, retryable_status_codes=(500, 502, 503, 504, 429)):
    """
    Decorator to retry function calls with exponential backoff on API errors.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries (exponential backoff)
        retryable_status_codes: HTTP status codes that should trigger a retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    # Check if this is an HTTP error we should retry
                    error_msg = str(e)
                    should_retry = False

                    # Check for retryable status codes
                    for code in retryable_status_codes:
                        if f"HTTP {code}" in error_msg:
                            should_retry = True
                            break

                    if not should_retry or attempt == max_retries:
                        # Not a retryable error, or we've exhausted retries
                        raise

                    # Log retry attempt
                    print(f"\n⚠️  API Error (attempt {attempt + 1}/{max_retries}): {error_msg.splitlines()[0]}")
                    print(f"   Retrying in {delay} seconds...")

                    time.sleep(delay)
                    delay *= backoff_factor  # Exponential backoff
                    last_exception = e
                except Exception as e:
                    # Non-RuntimeError exceptions are not retried
                    raise

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def parse_arguments():
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description='Sora 2 Extended Video Generation with AI Planning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses .env for API key)
  python src/sora_extend.py --prompt "Futuristic city tour" --segments 3

  # Full custom configuration
  python src/sora_extend.py \\
    --prompt "iPhone 19 intro" \\
    --segments 5 \\
    --duration 8 \\
    --output ./custom/path \\
    --planner-model gpt-4o \\
    --sora-model sora-2-pro

  # Docker usage
  docker exec sora-extend python src/sora_extend.py --prompt "..." --segments 2
        """
    )

    # Primary arguments
    parser.add_argument(
        '-p', '--prompt',
        dest='base_prompt',
        help='Base prompt for video generation (required if not in .env)'
    )

    parser.add_argument(
        '-s', '--segments',
        dest='num_generations',
        type=int,
        help='Number of video segments to generate (default: from .env or 2)'
    )

    parser.add_argument(
        '-d', '--duration',
        dest='seconds_per_segment',
        type=int,
        choices=[4, 8, 12],
        help='Duration per segment in seconds: 4, 8, or 12 (default: from .env or 8)'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory for generated videos (default: from .env or ./output/sora_ai_planned_chain)'
    )

    # API Configuration
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument(
        '--api-key',
        dest='openai_api_key',
        help='OpenAI API key (default: from .env or OPENAI_API_KEY env var)'
    )

    api_group.add_argument(
        '--api-base',
        dest='api_base',
        help='OpenAI API base URL (default: https://api.openai.com/v1)'
    )

    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--planner-model',
        dest='planner_model',
        help='AI model for planning segments (default: from .env or gpt-5)'
    )

    model_group.add_argument(
        '--sora-model',
        dest='sora_model',
        choices=['sora-2', 'sora-2-pro'],
        help='Sora model variant (default: from .env or sora-2)'
    )

    model_group.add_argument(
        '--size',
        dest='size',
        help='Video resolution, e.g., 1280x720, 1920x1080 (default: 1280x720)'
    )

    # Advanced Options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--work-dir',
        dest='work_dir',
        help='Working directory for operations (changes cwd before execution)'
    )

    advanced_group.add_argument(
        '--poll-interval',
        dest='poll_interval_sec',
        type=int,
        help='Polling interval in seconds for job status (default: 2)'
    )

    advanced_group.add_argument(
        '--no-env',
        dest='skip_env',
        action='store_true',
        help='Skip loading .env file (use only CLI args and system env vars)'
    )

    advanced_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    advanced_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without generating videos'
    )

    # Voiceover Options
    voiceover_group = parser.add_argument_group('Voiceover Options')
    voiceover_group.add_argument(
        '--enable-voiceover',
        action='store_true',
        help='Enable voiceover overlay using ElevenLabs (requires ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env)'
    )

    voiceover_group.add_argument(
        '--voiceover-text',
        dest='voiceover_text',
        help='Custom text for voiceover (default: uses base prompt)'
    )

    return parser.parse_args()


def get_config_value(arg_value, env_var_name, default=None, required=False):
    """
    Get configuration value with priority: CLI args > env vars > default

    Args:
        arg_value: Value from command-line argument
        env_var_name: Environment variable name
        default: Default value if not found
        required: Raise error if not found

    Returns:
        Configuration value
    """
    value = arg_value or os.environ.get(env_var_name) or default

    if required and not value:
        raise ValueError(
            f"{env_var_name} is required. "
            f"Provide via --{env_var_name.lower().replace('_', '-')} argument or .env file"
        )

    return value


# Load environment variables from .env file (unless --no-env is used)
# This is called conditionally in main() now


class SoraExtender:
    """Handles extended video generation using Sora 2 with AI-planned segments."""

    def __init__(self, args=None):
        """
        Initialize the Sora Extender with configuration.
        Priority: CLI arguments > environment variables > defaults

        Args:
            args: Optional argparse Namespace with command-line arguments
        """
        # API Configuration (with priority handling)
        self.api_key = get_config_value(
            args.openai_api_key if args else None,
            "OPENAI_API_KEY",
            required=True
        )

        self.client = OpenAI(api_key=self.api_key)

        self.api_base = get_config_value(
            args.api_base if args else None,
            "API_BASE",
            default="https://api.openai.com/v1"
        )

        self.headers_auth = {"Authorization": f"Bearer {self.api_key}"}

        # Model Configuration
        self.planner_model = get_config_value(
            args.planner_model if args else None,
            "PLANNER_MODEL",
            default="gpt-5"
        )

        self.sora_model = get_config_value(
            args.sora_model if args else None,
            "SORA_MODEL",
            default="sora-2"
        )

        self.size = get_config_value(
            args.size if args else None,
            "SIZE",
            default="1280x720"
        )

        # Generation Configuration
        self.base_prompt = get_config_value(
            args.base_prompt if args else None,
            "BASE_PROMPT",
            required=True
        )

        self.seconds_per_segment = int(get_config_value(
            args.seconds_per_segment if args else None,
            "SECONDS_PER_SEGMENT",
            default="8"
        ))

        self.num_generations = int(get_config_value(
            args.num_generations if args else None,
            "NUM_GENERATIONS",
            default="2"
        ))

        self.poll_interval = int(get_config_value(
            args.poll_interval_sec if args else None,
            "POLL_INTERVAL_SEC",
            default="2"
        ))

        # Output Configuration
        output_dir = get_config_value(
            args.output_dir if args else None,
            "OUTPUT_DIR",
            default="/app/output/sora_ai_planned_chain"
        )
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Voiceover Configuration (optional)
        self.enable_voiceover = args.enable_voiceover if args else False
        self.voiceover_text = None
        self.voiceover_orchestrator = None

        if self.enable_voiceover:
            if not VOICEOVER_AVAILABLE:
                raise RuntimeError(
                    "Voiceover components not available. "
                    "Ensure voiceover_orchestrator.py is in src/ directory."
                )

            # Get ElevenLabs credentials
            elevenlabs_api_key = get_config_value(
                None,  # No CLI arg for API key (use .env)
                "ELEVENLABS_API_KEY",
                required=True
            )
            elevenlabs_voice_id = get_config_value(
                None,  # No CLI arg for voice ID (use .env)
                "ELEVENLABS_VOICE_ID",
                required=True
            )

            # Get voiceover text (default to base prompt)
            self.voiceover_text = get_config_value(
                args.voiceover_text if args else None,
                "VOICEOVER_TEXT",
                default=self.base_prompt
            )

            # Initialize orchestrator
            self.voiceover_orchestrator = VoiceoverOrchestrator(
                api_key=elevenlabs_api_key,
                voice_id=elevenlabs_voice_id
            )

        print(f"Initialized Sora Extender:")
        print(f"  - Planner Model: {self.planner_model}")
        print(f"  - Sora Model: {self.sora_model}")
        print(f"  - Video Size: {self.size}")
        print(f"  - Seconds per Segment: {self.seconds_per_segment}")
        print(f"  - Number of Segments: {self.num_generations}")
        print(f"  - Output Directory: {self.out_dir}")
        print(f"  - Base Prompt: {self.base_prompt}")
        if self.enable_voiceover:
            print(f"  - Voiceover: ENABLED")
            print(f"  - Voiceover Text: {self.voiceover_text[:50]}...")
        print()

    @property
    def planner_system_instructions(self) -> str:
        """Returns the system instructions for the AI planner."""
        return r"""
You are a senior prompt director for Sora 2. Your job is to transform:
- a Base prompt (broad idea),
- a fixed generation length per segment (seconds),
- and a total number of generations (N),

into **N crystal-clear shot prompts** with **maximum continuity** across segments.

Rules:
1) Return **valid JSON** only. Structure:
   {
     "segments": [
       {
         "title": "Generation 1",
         "seconds": 6,
         "prompt": "<prompt block to send into Sora>"
       },
       ...
     ]
   }
   - `seconds` MUST equal the given generation length for ALL segments.
   - `prompt` should include a **Context** section for model guidance AND a **Prompt** line for the shot itself,
     exactly like in the example below.
2) Continuity:
   - Segment 1 starts fresh from the BASE PROMPT.
   - Segment k (k>1) must **begin exactly at the final frame** of segment k-1.
   - Maintain consistent visual style, tone, lighting, and subject identity unless explicitly told to change.
3) Safety & platform constraints:
   - Do not depict real people (including public figures) or copyrighted characters.
   - Avoid copyrighted music and avoid exact trademark/logos if policy disallows them; use brand-safe wording.
   - Keep content suitable for general audiences.
4) Output only JSON (no Markdown, no backticks).
5) Keep the **Context** lines inside the prompt text (they're for the AI, not visible).
6) Make the writing specific and cinematic; describe camera, lighting, motion, and subject focus succinctly.

Below is an **EXAMPLE (verbatim)** of exactly how to structure prompts with context and continuity:

Example:
Base prompt: "Intro video for the iPhone 19"
Generation length: 6 seconds each
Total generations: 3

Clearly defined prompts with maximum continuity and context:

### Generation 1:

<prompt>
First shot introducing the new iPhone 19. Initially, the screen is completely dark. The phone, positioned vertically and facing directly forward, emerges slowly and dramatically out of darkness, gradually illuminated from the center of the screen outward, showcasing a vibrant, colorful, dynamic wallpaper on its edge-to-edge glass display. The style is futuristic, sleek, and premium, appropriate for an official Apple product reveal.
<prompt>

---

### Generation 2:

<prompt>
Context (not visible in video, only for AI guidance):

* You are creating the second part of an official intro video for Apple's new iPhone 19.
* The previous 6-second scene ended with the phone facing directly forward, clearly displaying its vibrant front screen and colorful wallpaper.

Prompt: Second shot begins exactly from the final frame of the previous scene, showing the front of the iPhone 19 with its vibrant, colorful display clearly visible. Now, smoothly rotate the phone horizontally, turning it from the front to reveal the back side. Focus specifically on the advanced triple-lens camera module, clearly highlighting its premium materials, reflective metallic surfaces, and detailed lenses. Maintain consistent dramatic lighting, sleek visual style, and luxurious feel matching the official Apple product introduction theme.
</prompt>

---

### Generation 3:

<prompt>
Context (not visible in video, only for AI guidance):

* You are creating the third and final part of an official intro video for Apple's new iPhone 19.
* The previous 6-second scene ended clearly showing the back of the iPhone 19, focusing specifically on its advanced triple-lens camera module.

Prompt: Final shot begins exactly from the final frame of the previous scene, clearly displaying the back side of the iPhone 19, with special emphasis on the triple-lens camera module. Now, have a user's hand gently pick up the phone, naturally rotating it from the back to the front and bringing it upward toward their face. Clearly show the phone smoothly and quickly unlocking via Face ID recognition, transitioning immediately to a vibrant home screen filled with updated app icons. Finish the scene by subtly fading the home screen into the iconic Apple logo. Keep the visual style consistent, premium, and elegant, suitable for an official Apple product launch.
</prompt>

--

Notice how we broke up the initial prompt into multiple prompts that provide context and continuity so this all works seamlessly.
""".strip()

    def plan_prompts_with_ai(self) -> List[Dict]:
        """
        Use AI (GPT-5 or similar) to plan video segments with continuity.

        Returns:
            List of segment dictionaries with title, seconds, and prompt
        """
        print(f"Planning {self.num_generations} segments with {self.planner_model}...")

        user_input = f"""
BASE PROMPT: {self.base_prompt}

GENERATION LENGTH (seconds): {self.seconds_per_segment}
TOTAL GENERATIONS: {self.num_generations}

Return exactly {self.num_generations} segments.
""".strip()

        try:
            resp = self.client.responses.create(
                model=self.planner_model,
                instructions=self.planner_system_instructions,
                input=user_input,
            )

            text = getattr(resp, "output_text", None) or ""
            if not text:
                # Fallback: try to get text from response structure
                try:
                    text = json.dumps(resp.to_dict())
                except Exception:
                    raise RuntimeError("Planner returned no text; check PLANNER_MODEL.")

            # Extract JSON from response
            m = re.search(r'\{[\s\S]*\}', text)
            if not m:
                raise ValueError("Planner did not return JSON. Check model output.")

            data = json.loads(m.group(0))
            segments = data.get("segments", [])

            # Validate and enforce
            if len(segments) != self.num_generations:
                segments = segments[:self.num_generations]

            # Force correct duration
            for seg in segments:
                seg["seconds"] = int(self.seconds_per_segment)

            print(f"Successfully planned {len(segments)} segments\n")
            for i, seg in enumerate(segments, start=1):
                title = seg.get('title', '(untitled)')
                print(f"[{i:02d}] {seg['seconds']}s — {title}")
            print()

            return segments

        except Exception as e:
            raise RuntimeError(f"Failed to plan prompts: {e}")

    @staticmethod
    def guess_mime(path: Path) -> str:
        """Guess MIME type for a file."""
        mime = mimetypes.guess_type(str(path))[0]
        return mime or "application/octet-stream"

    def _dump_error(self, resp: requests.Response) -> str:
        """Format error response for debugging."""
        request_id = resp.headers.get("x-request-id", "<none>")
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return f"HTTP {resp.status_code} (request-id: {request_id})\n{body}"

    @retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2)
    def create_video(
        self,
        prompt: str,
        seconds: int,
        input_reference: Optional[Path] = None
    ) -> Dict:
        """
        Create a video generation job with Sora API (with automatic retries).

        Args:
            prompt: Video generation prompt
            seconds: Video duration
            input_reference: Optional path to reference image for continuity

        Returns:
            Job dictionary from API
        """
        files = {
            "model": (None, self.sora_model),
            "prompt": (None, prompt),
            "seconds": (None, str(seconds)),
        }

        if self.size:
            files["size"] = (None, self.size)

        if input_reference is not None:
            mime = self.guess_mime(input_reference)
            files["input_reference"] = (
                input_reference.name,
                open(input_reference, "rb"),
                mime
            )

        url = f"{self.api_base}/videos"
        resp = requests.post(url, headers=self.headers_auth, files=files, timeout=300)

        if resp.status_code >= 400:
            raise RuntimeError(f"Create video failed:\n{self._dump_error(resp)}")

        return resp.json()

    @retry_with_backoff(max_retries=5, initial_delay=2, backoff_factor=2)
    def retrieve_video(self, video_id: str) -> Dict:
        """Retrieve video job status with automatic retries on server errors."""
        url = f"{self.api_base}/videos/{video_id}"
        resp = requests.get(url, headers=self.headers_auth, timeout=60)

        if resp.status_code >= 400:
            raise RuntimeError(f"Retrieve video failed:\n{self._dump_error(resp)}")

        return resp.json()

    @retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2)
    def download_video_content(
        self,
        video_id: str,
        out_path: Path,
        variant: str = "video"
    ) -> Path:
        """Download completed video content (with automatic retries)."""
        url = f"{self.api_base}/videos/{video_id}/content"
        params = {"variant": variant}

        with requests.get(
            url,
            headers=self.headers_auth,
            params=params,
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"Download failed:\n{self._dump_error(resp)}")

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return out_path

    def poll_until_complete(self, job: Dict) -> Dict:
        """
        Poll video generation job until complete.

        Args:
            job: Initial job dictionary

        Returns:
            Completed job dictionary
        """
        video = job
        video_id = video["id"]

        def progress_bar(pct: float, width: int = 30) -> str:
            """Generate a text progress bar."""
            filled = int(max(0.0, min(100.0, pct)) / 100 * width)
            return "=" * filled + "-" * (width - filled)

        while video.get("status") in ("queued", "in_progress"):
            pct = float(video.get("progress", 0) or 0)
            status_text = "Queued" if video["status"] == "queued" else "Processing"
            print(f"\r{status_text}: [{progress_bar(pct)}] {pct:5.1f}%", end="", flush=True)

            time.sleep(self.poll_interval)
            video = self.retrieve_video(video_id)

        print()  # Newline after progress bar

        if video.get("status") != "completed":
            error_msg = (video.get("error") or {}).get("message", f"Job {video_id} failed")
            raise RuntimeError(error_msg)

        return video

    @staticmethod
    def extract_last_frame(video_path: Path, out_image_path: Path) -> Path:
        """
        Extract the last frame from a video for continuity reference.

        Args:
            video_path: Path to video file
            out_image_path: Path to save last frame image

        Returns:
            Path to saved image
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        success, frame = False, None

        # Try to seek to last frame
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            success, frame = cap.read()

        # Fallback: read through entire video
        if not success or frame is None:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frame = f
                success = True

        cap.release()

        if not success or frame is None:
            raise RuntimeError(f"Could not read last frame from {video_path}")

        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_image_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write {out_image_path}")

        return out_image_path

    def chain_generate_sora(self, segments: List[Dict]) -> List[Path]:
        """
        Generate video segments with continuity using planned prompts.

        Args:
            segments: List of segment dictionaries from planner

        Returns:
            List of paths to generated video segments
        """
        input_ref = None
        segment_paths = []

        for i, seg in enumerate(segments, start=1):
            seconds = int(seg["seconds"])
            prompt = seg["prompt"]
            title = seg.get("title", f"Segment {i}")

            print(f"\n{'='*70}")
            print(f"Generating Segment {i}/{len(segments)} — {seconds}s")
            print(f"Title: {title}")
            print(f"{'='*70}")

            # Create video generation job
            job = self.create_video(
                prompt=prompt,
                seconds=seconds,
                input_reference=input_ref
            )
            print(f"Started job: {job['id']} | status: {job['status']}")

            # Poll until complete
            completed = self.poll_until_complete(job)

            # Download video
            seg_path = self.out_dir / f"segment_{i:02d}.mp4"
            self.download_video_content(completed["id"], seg_path, variant="video")
            print(f"Saved: {seg_path}")
            segment_paths.append(seg_path)

            # Extract last frame for next segment's continuity
            frame_path = self.out_dir / f"segment_{i:02d}_last.jpg"
            self.extract_last_frame(seg_path, frame_path)
            print(f"Extracted last frame: {frame_path}")
            input_ref = frame_path

        return segment_paths

    def concatenate_segments(
        self,
        segment_paths: List[Path],
        out_path: Path
    ) -> Path:
        """
        Concatenate video segments into a single video.

        Args:
            segment_paths: List of paths to video segments
            out_path: Output path for combined video

        Returns:
            Path to combined video
        """
        print(f"\n{'='*70}")
        print("Concatenating segments...")
        print(f"{'='*70}")

        clips = [VideoFileClip(str(p)) for p in segment_paths]
        target_fps = clips[0].fps or 24

        result = concatenate_videoclips(clips, method="compose")
        result.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=target_fps,
            preset="medium",
            threads=0
        )

        for clip in clips:
            clip.close()

        return out_path

    def run(self):
        """Execute the complete video extension pipeline."""
        print("\n" + "="*70)
        print("SORA VIDEO EXTENSION PIPELINE")
        print("="*70 + "\n")

        # Step 1: Plan segments with AI
        print("STEP 1: Planning video segments with AI")
        print("-"*70)
        segments = self.plan_prompts_with_ai()

        # Step 2: Generate segments with Sora
        print("\nSTEP 2: Generating video segments with Sora")
        print("-"*70)
        segment_paths = self.chain_generate_sora(segments)

        # Step 3: Concatenate segments
        print("\nSTEP 3: Concatenating segments")
        print("-"*70)
        final_path = self.out_dir / "combined.mp4"
        self.concatenate_segments(segment_paths, final_path)

        # Step 4 (Optional): Overlay voiceover
        if self.enable_voiceover:
            print("\nSTEP 4: Overlaying voiceover")
            print("-"*70)
            print(f"Generating voiceover from text: {self.voiceover_text[:80]}...")

            voiceover_path = self.out_dir / "combined_with_voiceover.mp4"

            try:
                self.voiceover_orchestrator.process(
                    video_path=str(final_path),
                    text=self.voiceover_text,
                    output_path=str(voiceover_path)
                )
                print(f"Voiceover overlay complete: {voiceover_path}")

                # Update final path to the voiceover version
                final_path = voiceover_path

            except Exception as e:
                print(f"⚠️  Voiceover overlay failed: {e}")
                print(f"   Continuing with original video without voiceover.")
                # Keep original final_path

        # Done!
        print(f"\n{'='*70}")
        print("COMPLETE!")
        print(f"{'='*70}")
        print(f"Final video: {final_path}")
        print(f"Total duration: {self.seconds_per_segment * self.num_generations} seconds")
        print(f"Number of segments: {len(segment_paths)}")
        if self.enable_voiceover:
            print(f"Voiceover: {'ENABLED' if str(final_path).endswith('_with_voiceover.mp4') else 'FAILED (using original)'}")
        print(f"Output directory: {self.out_dir}")
        print(f"{'='*70}\n")


def main():
    """Main entry point with CLI support."""
    # Parse command-line arguments
    args = parse_arguments()

    # Load .env file unless --no-env is specified
    if not args.skip_env:
        load_dotenv()

    # Handle work directory change
    if args.work_dir:
        os.chdir(args.work_dir)
        print(f"Changed working directory to: {args.work_dir}\n")

    try:
        extender = SoraExtender(args)

        # Dry run mode - show configuration without generating
        if args.dry_run:
            print("\n" + "="*70)
            print("DRY RUN MODE - Configuration Preview")
            print("="*70)
            print(f"  Base Prompt: {extender.base_prompt}")
            print(f"  Number of Segments: {extender.num_generations}")
            print(f"  Duration per Segment: {extender.seconds_per_segment}s")
            print(f"  Total Duration: {extender.num_generations * extender.seconds_per_segment}s")
            print(f"  Output Directory: {extender.out_dir}")
            print(f"  Planner Model: {extender.planner_model}")
            print(f"  Sora Model: {extender.sora_model}")
            print(f"  Video Size: {extender.size}")
            print(f"  API Base: {extender.api_base}")
            print("="*70)
            print("\nNo videos will be generated in dry-run mode.")
            print("Remove --dry-run flag to execute actual generation.")
            return

        # Run the video generation pipeline
        extender.run()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

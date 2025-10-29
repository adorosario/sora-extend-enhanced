#!/usr/bin/env python3
"""
Sora 2 AI-Planned Video Extension
Extends OpenAI's Sora 2 video generation beyond 12 seconds by chaining multiple segments
with intelligent planning and continuity.
"""

import os
import re
import json
import time
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict

import requests
import cv2
from moviepy import VideoFileClip, concatenate_videoclips
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SoraExtender:
    """Handles extended video generation using Sora 2 with AI-planned segments."""

    def __init__(self):
        """Initialize the Sora Extender with configuration from environment variables."""
        # API Configuration
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=self.api_key)
        self.api_base = os.environ.get("API_BASE", "https://api.openai.com/v1")
        self.headers_auth = {"Authorization": f"Bearer {self.api_key}"}

        # Model Configuration
        self.planner_model = os.environ.get("PLANNER_MODEL", "gpt-5")
        self.sora_model = os.environ.get("SORA_MODEL", "sora-2")
        self.size = os.environ.get("SIZE", "1280x720")

        # Generation Configuration
        self.base_prompt = os.environ.get("BASE_PROMPT")
        if not self.base_prompt:
            raise ValueError("BASE_PROMPT environment variable is required")

        self.seconds_per_segment = int(os.environ.get("SECONDS_PER_SEGMENT", "8"))
        self.num_generations = int(os.environ.get("NUM_GENERATIONS", "2"))
        self.poll_interval = int(os.environ.get("POLL_INTERVAL_SEC", "2"))

        # Output Configuration
        output_dir = os.environ.get("OUTPUT_DIR", "/app/output/sora_ai_planned_chain")
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized Sora Extender:")
        print(f"  - Planner Model: {self.planner_model}")
        print(f"  - Sora Model: {self.sora_model}")
        print(f"  - Video Size: {self.size}")
        print(f"  - Seconds per Segment: {self.seconds_per_segment}")
        print(f"  - Number of Segments: {self.num_generations}")
        print(f"  - Output Directory: {self.out_dir}")
        print(f"  - Base Prompt: {self.base_prompt}")
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

    def create_video(
        self,
        prompt: str,
        seconds: int,
        input_reference: Optional[Path] = None
    ) -> Dict:
        """
        Create a video generation job with Sora API.

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

    def retrieve_video(self, video_id: str) -> Dict:
        """Retrieve video job status."""
        url = f"{self.api_base}/videos/{video_id}"
        resp = requests.get(url, headers=self.headers_auth, timeout=60)

        if resp.status_code >= 400:
            raise RuntimeError(f"Retrieve video failed:\n{self._dump_error(resp)}")

        return resp.json()

    def download_video_content(
        self,
        video_id: str,
        out_path: Path,
        variant: str = "video"
    ) -> Path:
        """Download completed video content."""
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

        # Done!
        print(f"\n{'='*70}")
        print("COMPLETE!")
        print(f"{'='*70}")
        print(f"Final video: {final_path}")
        print(f"Total duration: {self.seconds_per_segment * self.num_generations} seconds")
        print(f"Number of segments: {len(segment_paths)}")
        print(f"Output directory: {self.out_dir}")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    try:
        extender = SoraExtender()
        extender.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test script to apply voiceover to existing video."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voiceover_orchestrator import VoiceoverOrchestrator

def main():
    # Get credentials from environment
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")

    if not api_key or not voice_id:
        print("ERROR: ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")
        sys.exit(1)

    # Input video (the successful UGC video)
    input_video = Path("/app/output/spider_mites_ugc_v2/combined.mp4")

    if not input_video.exists():
        print(f"ERROR: Input video not found: {input_video}")
        sys.exit(1)

    # Output video
    output_video = Path("/app/output/spider_mites_ugc_v2/combined_with_voiceover.mp4")

    # Voiceover text (HGTV-style narration about spider mites)
    voiceover_text = """Think your houseplants are just having a bad day? Those tiny yellow dots on the leaves might be telling you something more serious. I'm going to show you how to spot spider mites before they destroy your favorite plants—and the simple organic solution that actually works.

Look closely at the underside of your leaves. See those tiny specks? That's stippling—the telltale sign of spider mites feeding on your plant's chlorophyll. They're nearly invisible to the naked eye, but a simple magnifying lens reveals the truth. These pests thrive in hot, dry conditions, spreading faster than you'd think.

Here's the good news: you don't need harsh chemicals. Organic neem oil is nature's answer to spider mites. Apply it gently to the undersides of leaves—that's where they hide. The oil disrupts their life cycle while being completely safe for your home.

Pair this with increasing humidity around your plants, and you're creating an environment where mites can't survive, but your plants will thrive. Remember, healthy plants are your best defense. Keep them well-watered, check regularly, and catch problems early. Your indoor garden will thank you."""

    print("=" * 70)
    print("VOICEOVER TEST - End-to-End ElevenLabs Integration")
    print("=" * 70)
    print(f"\nInput video:  {input_video}")
    print(f"Output video: {output_video}")
    print(f"Voiceover text: {len(voiceover_text)} characters (~{len(voiceover_text.split())} words)")
    print("\nInitializing VoiceoverOrchestrator...")

    # Create orchestrator
    orchestrator = VoiceoverOrchestrator(
        api_key=api_key,
        voice_id=voice_id
    )

    print("\nProcessing video with voiceover...")
    print("This will:")
    print("  1. Generate voiceover audio from text (ElevenLabs API)")
    print("  2. Strip existing audio from video (MoviePy)")
    print("  3. Overlay new voiceover onto video (MoviePy)")
    print("  4. Clean up temporary files")
    print()

    try:
        result = orchestrator.process(
            video_path=str(input_video),
            text=voiceover_text,
            output_path=str(output_video)
        )

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nOutput video created: {result}")
        print(f"File size: {Path(result).stat().st_size / 1024 / 1024:.2f} MB")
        print("\nYou can now compare:")
        print(f"  - Original (Sora audio):     {input_video}")
        print(f"  - With voiceover (ElevenLabs): {output_video}")

    except Exception as e:
        print("\n" + "=" * 70)
        print("FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

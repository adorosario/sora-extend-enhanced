"""ScriptGenerator component for converting LinkedIn posts to video scripts.

This module provides functionality to convert LinkedIn post text into
natural-sounding speaking scripts optimized for video voiceover.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI


class ScriptGenerator:
    """Converts LinkedIn posts into video scripts with theme detection."""

    def __init__(self, api_key: str, templates_path: Optional[str] = None):
        """Initialize ScriptGenerator with API credentials.

        Args:
            api_key: OpenAI API key
            templates_path: Path to studio templates JSON file
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

        # Load studio templates
        if templates_path is None:
            templates_path = Path(__file__).parent.parent / "prompts" / "studio_templates.json"

        with open(templates_path, 'r') as f:
            self.templates = json.load(f)

    def generate_script(
        self,
        linkedin_post: str,
        target_duration: int = 60,
        model: str = "gpt-4o"
    ) -> Dict[str, any]:
        """Generate speaking script from LinkedIn post.

        Args:
            linkedin_post: The LinkedIn post text
            target_duration: Target video duration in seconds (default: 60)
            model: OpenAI model to use (default: gpt-4o, can use gpt-5 if available)

        Returns:
            dict: {
                'script': str,
                'visual_theme': str,
                'topic': str,
                'estimated_duration': int,
                'word_count': int
            }

        Raises:
            ValueError: If linkedin_post is empty
            RuntimeError: If script generation fails
        """
        if not linkedin_post or not linkedin_post.strip():
            raise ValueError("LinkedIn post cannot be empty")

        # Calculate target word count (2.5-3 words per second for natural speech)
        # For 12 seconds: 2.5 * 12 = 30 words (conservative)
        # Hard maximum: 2.8 * 12 = 33.6 words (absolute ceiling)
        target_words = int(target_duration * 2.5)  # Conservative target
        max_words = int(target_duration * 2.8)     # Hard ceiling

        # Build theme descriptions for the AI
        theme_descriptions = {}
        for theme_key, theme_data in self.templates.items():
            theme_descriptions[theme_key] = {
                "name": theme_data["name"],
                "keywords": theme_data["themes"]
            }

        # Create the prompt for GPT
        system_prompt = f"""You are an expert content strategist who converts LinkedIn posts into engaging video scripts.

Your task is to:
1. Transform written LinkedIn posts into natural, conversational speaking scripts
2. Maintain the core message and key insights
3. Optimize for spoken delivery (short sentences, natural pauses, conversational tone)
4. Extract the main topic/theme from the post (2-4 words like "quality and excellence" or "startup growth" or "leadership wisdom")
5. Create a compelling hook
6. Select the most appropriate visual theme based on content

CRITICAL REQUIREMENTS:
- Script MUST be exactly {target_words}-{max_words} words (STRICT LIMIT)
- Script MUST fit in exactly {target_duration} seconds when spoken at natural pace
- Use ultra-concise, punchy sentences
- Every word must count - no fluff
- Use contractions (I'm, don't, it's) for brevity
- Write as if you're texting, not essaying

Response format (JSON):
{{
  "script": "The complete speaking script...",
  "topic": "2-4 word topic extracted from post",
  "visual_theme": "theme_key",
  "theme_rationale": "Brief explanation",
  "estimated_duration": {target_duration},
  "word_count": (must be between {target_words} and {max_words})
}}"""

        user_prompt = f"""LinkedIn Post:
{linkedin_post}

CRITICAL CONSTRAINTS:
- Target Duration: {target_duration} seconds (EXACT)
- Target Word Count: {target_words} words (MINIMUM)
- Maximum Word Count: {max_words} words (ABSOLUTE CEILING - DO NOT EXCEED)

Available Visual Themes:
{json.dumps(theme_descriptions, indent=2)}

Convert this LinkedIn post into an ultra-concise {target_duration}-second video script. Every word must earn its place. Extract the core topic in 2-4 words."""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            # Validate visual_theme
            if result.get('visual_theme') not in self.templates:
                # Default to classic_studio if invalid theme
                result['visual_theme'] = 'classic_studio'

            # Validate word count
            actual_word_count = len(result['script'].split())
            if actual_word_count > max_words:
                # Trim script to fit within max_words
                words = result['script'].split()
                result['script'] = ' '.join(words[:max_words])
                actual_word_count = max_words

            # Extract topic (or default from content)
            topic = result.get('topic', 'the topic')

            return {
                'script': result['script'],
                'topic': topic,
                'visual_theme': result['visual_theme'],
                'estimated_duration': result.get('estimated_duration', target_duration),
                'word_count': actual_word_count,
                'theme_rationale': result.get('theme_rationale', '')
            }

        except Exception as e:
            raise RuntimeError(f"Failed to generate script: {e}")

    def get_studio_prompt(self, theme: str, topic: str = "the topic") -> str:
        """Get Sora prompt for a specific studio theme with topic inserted.

        Args:
            theme: Theme key (e.g., 'classic_studio', 'modern_studio')
            topic: Topic to insert into prompt (e.g., "quality and excellence")

        Returns:
            str: Sora prompt for the studio B-roll with topic filled in

        Raises:
            ValueError: If theme is not found
        """
        if theme not in self.templates:
            raise ValueError(f"Unknown theme: {theme}. Available: {list(self.templates.keys())}")

        prompt_template = self.templates[theme]['prompt']
        # Replace {topic} placeholder with actual topic
        return prompt_template.replace('{topic}', topic)

    def save_script(self, script_data: Dict, output_path: str) -> str:
        """Save script data to JSON file.

        Args:
            script_data: Script data dictionary from generate_script()
            output_path: Path to save JSON file

        Returns:
            str: Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(script_data, f, indent=2)

        return str(output_path)


# CLI interface for testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Generate video script from LinkedIn post")
    parser.add_argument("--input", required=True, help="Path to LinkedIn post text file")
    parser.add_argument("--output", help="Path to save script JSON (default: script.json)")
    parser.add_argument("--duration", type=int, default=60, help="Target duration in seconds (default: 60)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Read LinkedIn post
    try:
        with open(args.input, 'r') as f:
            linkedin_post = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Generate script
    print(f"Generating script from {args.input}...")
    generator = ScriptGenerator(api_key=api_key)

    try:
        result = generator.generate_script(
            linkedin_post=linkedin_post,
            target_duration=args.duration,
            model=args.model
        )

        print(f"\n✓ Script generated successfully!")
        print(f"  Visual Theme: {result['visual_theme']}")
        print(f"  Word Count: {result['word_count']}")
        print(f"  Estimated Duration: {result['estimated_duration']}s")
        print(f"\nScript:\n{result['script']}")

        if result.get('theme_rationale'):
            print(f"\nTheme Rationale: {result['theme_rationale']}")

        # Save to file if output specified
        if args.output:
            output_file = generator.save_script(result, args.output)
            print(f"\n✓ Script saved to: {output_file}")

    except Exception as e:
        print(f"Error generating script: {e}")
        sys.exit(1)

# Sora Extend Enhanced

[![Original by Matt Shumer](https://img.shields.io/badge/Original-@mattshumer_-blue)](https://github.com/mshumer/sora-extend) [![Follow on X](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mshumer/sora-extend/blob/main/Sora_Extend.ipynb)

**Enhanced version with Docker support and CLI arguments for extended Sora 2 video generation.**

> **Credits:** This is an enhanced version of [sora-extend](https://github.com/mshumer/sora-extend) by [Matt Shumer](https://x.com/mattshumer_). All core video generation logic and AI planning approach belongs to the original author.

## What's New in This Fork

- ✅ **Docker Containerization**: Run with zero host dependencies
- ✅ **CLI Arguments**: Command-line interface for headless operation
- ✅ **Python Script**: Converted from Jupyter notebook for production use
- ✅ **Better Configuration**: Comprehensive .env.example with documentation
- ✅ **Safety Features**: Manual execution to prevent accidental API charges

---

**Seamlessly generate extended Sora 2 videos beyond OpenAI's 12-second limit.**

OpenAI’s Sora video generation model currently restricts output to 12-second clips. By leveraging the final frame of each generation as context for the next, and intelligently breaking down your prompt into coherent segments that mesh well, Sora Extend enables the creation of high-quality, extended-duration videos with continuity.

---

## How it Works

1. **Prompt Deconstruction**

   * Your initial prompt is intelligently broken down into smaller, coherent segments suitable for Sora 2’s native generation limits, with additional context that allows each subsequent prompt to have a sense of what happened before it.

2. **Sequential Video Generation**

   * Each prompt segment is independently processed by Sora 2, sequentially, generating video clips that align smoothly both visually and thematically. The final frame of each generated clip is captured and fed into the subsequent generation step as contextual input, helping with visual consistency.

3. **Concatenation**

   * Generated video segments are concatenated automatically, resulting in a single continuous video output without noticeable transitions or interruptions.

---

## Docker Setup (Recommended)

Run Sora Extend with zero host dependencies using Docker.

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key with access to:
  - Sora 2 API
  - GPT-5 (or alternative planning model like GPT-4o)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mshumer/sora-extend.git
cd sora-extend

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API key and video parameters

# 3. Start the container (builds if needed)
docker-compose up -d

# 4. Run the video generation (when you're ready)
docker exec sora-extend python src/sora_extend.py

# 5. Find your video in ./output/sora_ai_planned_chain/combined.mp4
```

**Note:** The container doesn't auto-run the script to avoid unexpected API charges. You manually execute it when ready.

### Configuration

Edit `.env` to configure your video generation:

#### Required Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `BASE_PROMPT` - Description of the video you want to create
- `SECONDS_PER_SEGMENT` - Duration of each segment (4, 8, or 12 seconds)
- `NUM_GENERATIONS` - Number of segments to generate

#### Example Configuration

```bash
OPENAI_API_KEY=sk-proj-your-key-here
BASE_PROMPT="Gameplay footage of a game releasing in 2027, a car driving through a futuristic city"
SECONDS_PER_SEGMENT=8
NUM_GENERATIONS=5
```

This generates a 40-second video (8 seconds × 5 segments).

#### Optional Variables

- `PLANNER_MODEL` - AI model for planning (default: `gpt-5`)
- `SORA_MODEL` - Sora model variant (default: `sora-2`, or `sora-2-pro`)
- `SIZE` - Video resolution (default: `1280x720`)
- `POLL_INTERVAL_SEC` - Status check interval (default: `2`)

See `.env.example` for all available options and detailed descriptions.

### Output

Generated videos are saved to `./output/sora_ai_planned_chain/`:
- `segment_01.mp4`, `segment_02.mp4`, etc. - Individual segments
- `segment_01_last.jpg`, etc. - Final frames for continuity
- `combined.mp4` - Final concatenated video

---

## CLI Usage (Headless Operation)

The script now supports command-line arguments for headless operation and automation!

### Quick Examples

```bash
# Basic: Override prompt (uses .env for other settings)
docker exec sora-extend python src/sora_extend.py \
  --prompt "Futuristic city tour with flying cars"

# Custom segments and duration
docker exec sora-extend python src/sora_extend.py \
  --prompt "iPhone 19 product launch" \
  --segments 5 \
  --duration 8

# Full custom configuration
docker exec sora-extend python src/sora_extend.py \
  --prompt "Cinematic gameplay footage" \
  --segments 3 \
  --duration 12 \
  --output /app/output/custom_project \
  --planner-model gpt-4o \
  --sora-model sora-2-pro
```

### All Available Arguments

**Primary Options:**
- `-p, --prompt` - Base prompt for video generation
- `-s, --segments` - Number of video segments to generate
- `-d, --duration` - Duration per segment: 4, 8, or 12 seconds
- `-o, --output` - Output directory path

**API Configuration:**
- `--api-key` - OpenAI API key (overrides .env)
- `--api-base` - API base URL (default: https://api.openai.com/v1)

**Model Configuration:**
- `--planner-model` - AI planning model (default: gpt-5)
- `--sora-model` - Sora variant: sora-2 or sora-2-pro
- `--size` - Video resolution (default: 1280x720)

**Advanced:**
- `--work-dir` - Change working directory before execution
- `--poll-interval` - Status polling interval in seconds
- `--dry-run` - Preview configuration without generating
- `--no-env` - Skip .env loading (CLI args only)
- `-v, --verbose` - Enable verbose error output

### Configuration Priority

Settings are applied in this order (highest to lowest priority):
1. **CLI arguments** (e.g., `--prompt "..."`)
2. **Environment variables** (from `.env` file)
3. **Default values** (hardcoded fallbacks)

### Dry Run Mode

Test your configuration without API calls:

```bash
docker exec sora-extend python src/sora_extend.py \
  --prompt "Test prompt" \
  --segments 3 \
  --duration 12 \
  --dry-run
```

Output:
```
======================================================================
DRY RUN MODE - Configuration Preview
======================================================================
  Base Prompt: Test prompt
  Number of Segments: 3
  Duration per Segment: 12s
  Total Duration: 36s
  Output Directory: /app/output/sora_ai_planned_chain
  Planner Model: gpt-5
  Sora Model: sora-2
  Video Size: 1280x720
======================================================================
```

### Help

View all options:
```bash
docker exec sora-extend python src/sora_extend.py --help
```

---

### Troubleshooting

**Issue**: "OPENAI_API_KEY environment variable is required"
- Make sure you've copied `.env.example` to `.env` and added your API key

**Issue**: Video generation fails or times out
- Check your OpenAI API quota and rate limits
- Verify you have access to Sora 2 API
- Try reducing `NUM_GENERATIONS` for testing

**Issue**: "Planner did not return JSON"
- If you don't have GPT-5 access, set `PLANNER_MODEL=gpt-4o` in `.env`

**Issue**: Output directory is empty
- Check Docker logs: `docker-compose logs`
- Ensure the container has write permissions to `./output`

---

## Original Notebook Usage

For the original Jupyter notebook experience, see [Sora_Extend.ipynb](Sora_Extend.ipynb) or try it in [Google Colab](https://colab.research.google.com/github/mshumer/sora-extend/blob/main/Sora_Extend.ipynb).

---

## Credits & License

### Original Work

This enhanced version is based on [sora-extend](https://github.com/mshumer/sora-extend) by **[Matt Shumer](https://x.com/mattshumer_)**.

- All core video generation logic, AI planning approach, and prompt engineering belongs to the original author
- Original repository: https://github.com/mshumer/sora-extend
- Follow Matt on X: [@mattshumer_](https://x.com/mattshumer_)

[Be the first to know when Matt publishes new AI builds + demos!](https://tally.so/r/w2M17p)

### Enhancements

This fork adds:
- Docker containerization for zero-dependency execution
- Command-line interface with argument support
- Production-ready Python script conversion
- Improved configuration management
- Safety features (manual execution, no auto-run)

Enhanced by: [@adorosario](https://github.com/adorosario)

### License

Please refer to the [original repository](https://github.com/mshumer/sora-extend) for license terms.

---

Generate long-form AI videos effortlessly with Sora Extend.

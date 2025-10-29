# Sora Extend

[![Follow on X](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mshumer/sora-extend/blob/main/Sora_Extend.ipynb)

[Be the first to know when I publish new AI builds + demos!](https://tally.so/r/w2M17p)

**Seamlessly generate extended Sora 2 videos beyond OpenAI’s 12-second limit.**

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

For the original Jupyter notebook experience, see [Sora_Extend.ipynb](Sora_Extend.ipynb).

---

Generate long-form AI videos effortlessly with Sora Extend.

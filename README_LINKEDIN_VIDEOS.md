# LinkedIn Post-to-Video Generator

Automatically convert LinkedIn posts into professional UGC videos with studio B-roll and voiceover narration.

## 🎯 Overview

This system transforms your LinkedIn thought-leadership posts into engaging 30-60 second videos featuring:
- **Cinematic studio B-roll** (generated via Sora 2)
- **Professional voiceover** (your ElevenLabs voice)
- **Dynamic visual themes** (adapts to post content)
- **Full automation** (paste post, get video)

## 💰 Cost per Video

- **60-second video**: ~$3.30
  - GPT-4o script generation: $0.10
  - Sora 2 studio B-roll: $3.00
  - ElevenLabs voiceover: $0.20

- **30-second video**: ~$1.70
  - Sora 2: $1.50
  - ElevenLabs: $0.10
  - GPT-4o: $0.10

## 🏗️ Architecture

```
LinkedIn Post (.txt)
       ↓
[Script Generator] (GPT-4o, ~10s)
       ↓
   { script, theme, duration }
       ↓
   ┌───────────────┴───────────────┐
   ↓                               ↓
[Sora Studio B-Roll]          [ElevenLabs Voiceover]
  (3-5 min)                      (30s)
   ↓                               ↓
studio_broll.mp4              voiceover.mp3
   └───────────────┬───────────────┘
                   ↓
         [MoviePy Compositor] (30s)
                   ↓
          final_video.mp4
```

**Total time**: 4-6 minutes per video

## 📋 Prerequisites

1. **OpenAI API Key** with Sora 2 access
2. **ElevenLabs API Key** and Voice ID
3. **Docker & Docker Compose** installed
4. **.env file** configured (see below)

## 🚀 Quick Start

### 1. Configure Environment

Your `.env` file should have:

```bash
# OpenAI (for GPT and Sora)
OPENAI_API_KEY=sk-proj-...

# ElevenLabs (for voiceover)
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=XrExE9yKIg1WjnnlVkGX  # Your cloned voice ID
```

### 2. Create LinkedIn Post File

Save your LinkedIn post as a `.txt` file:

```bash
cat > my_post.txt << 'EOF'
Steve Jobs on Quality

Most companies focus on shipping fast. Get it out there, iterate later.

But Jobs understood something deeper: first impressions are permanent...
EOF
```

### 3. Generate Video

```bash
# Build Docker image (first time only)
docker compose build

# Generate video
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --name steve_jobs_quality \
  --duration 60
```

### 4. Get Your Video

Output will be in:
```
./output/linkedin_videos/steve_jobs_quality/
├── steve_jobs_quality.mp4    ← FINAL VIDEO
├── script.json               ← Generated script & metadata
├── studio_broll.mp4          ← Sora B-roll footage
└── voiceover.mp3             ← ElevenLabs narration
```

## 🎨 Visual Themes

The system automatically selects the best visual theme based on your post content:

### Classic Studio
**Best for**: Leadership, wisdom, quality, timeless topics
- Warm amber lighting
- Vintage microphone on boom arm
- Rich wooden desk with bookshelf
- Cozy, professional atmosphere

**Keywords detected**: leadership, quality, wisdom, experience, principles, values

### Modern Studio
**Best for**: Tech, innovation, startups, digital topics
- Cool blue LED accent lighting
- Minimalist glass desk
- Contemporary microphone setup
- Clean, tech-forward aesthetic

**Keywords detected**: technology, innovation, future, startup, digital, AI, software, product

### Intimate Studio
**Best for**: Personal stories, career journeys, lessons learned
- Close-up studio details
- Warm natural window light
- Comfortable, lived-in space
- Shallow focus, golden hour lighting

**Keywords detected**: personal, story, journey, reflection, career, lesson, challenge, growth

### Energetic Studio
**Best for**: Motivation, calls to action, momentum
- Dynamic camera movement
- Dramatic lighting with contrast
- Modern professional equipment
- High contrast, vibrant colors

**Keywords detected**: motivation, inspiration, action, energy, momentum, hustle, success

## 🔧 Advanced Usage

### Test Script Generation Only

Want to see the script before generating video?

```bash
docker compose run --rm sora-extend python src/script_generator.py \
  --input my_post.txt \
  --output my_script.json \
  --duration 60
```

This will show you:
- Generated speaking script
- Selected visual theme
- Word count & estimated duration
- Theme selection rationale

### Custom Duration

Generate shorter or longer videos:

```bash
# 30-second video (cheaper)
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --duration 30

# 90-second video (more expensive)
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --duration 90
```

### Keep Intermediate Files

By default, intermediate files are kept. To remove them:

```bash
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --keep-intermediates false
```

### Custom Output Directory

```bash
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --output-dir ./my_videos
```

## 📊 Workflow Example

```bash
# 1. Save LinkedIn post
echo "Your LinkedIn post content here..." > posts/week1.txt

# 2. Generate video
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input posts/week1.txt \
  --name week1_video

# 3. Wait ~5 minutes...

# 4. Video ready!
ls output/linkedin_videos/week1_video/week1_video.mp4
```

## 🎬 Script Generation

The script generator (GPT-4o) converts your written post into natural speech:

**Optimizations**:
- Conversational tone (uses contractions, short sentences)
- Hook in first 5-10 seconds
- Strong call-to-action
- 2.5-3 words per second pacing
- Natural pauses and punctuation

**Example**:

**Input** (LinkedIn post):
```
Most companies focus on shipping fast.
But Jobs understood: first impressions are permanent.
```

**Output** (Speaking script):
```
Hey there! Let's talk about a lesson from Steve Jobs. You know,
most companies these days are all about speed. Get it out there
fast, fix it later. But Jobs? He saw things differently. He knew
that first impressions are permanent. You just can't patch your
way to excellence...
```

## 🛠️ Components

### 1. Script Generator (`src/script_generator.py`)
- Reads LinkedIn post
- Calls GPT-4o to generate speaking script
- Detects appropriate visual theme
- Optimizes for natural speech delivery

### 2. LinkedIn Video Generator (`src/linkedin_video_generator.py`)
- Orchestrates end-to-end pipeline
- Parallel generation (Sora + ElevenLabs)
- Video compositing with MoviePy
- Error handling & retry logic

### 3. Voiceover Generator (`src/voiceover_generator.py`)
- ElevenLabs text-to-speech
- Exponential backoff retry
- High-quality multilingual voice

### 4. Studio Templates (`prompts/studio_templates.json`)
- Theme definitions
- Sora prompts for each style
- Keyword mappings

## 📝 Tips for Best Results

### Writing LinkedIn Posts
- **Clear structure**: Hook → Main points → CTA
- **Conversational**: Write how you'd speak
- **Focused**: 1-2 key ideas per post
- **Authentic**: Personal stories resonate

### Video Quality
- **First 3 seconds matter**: Strong hook crucial
- **Keep it tight**: 30-60s is optimal for social
- **Consistent posting**: Weekly cadence builds audience
- **Test themes**: Different visuals for different topics

### Cost Optimization
- **Batch processing**: Generate multiple videos at once
- **30s sweet spot**: Cheaper, higher engagement
- **Review scripts**: Test script generation before full pipeline
- **Reuse B-roll**: Similar topics can share visual style

## 🐛 Troubleshooting

### "OpenAI API Key not set"
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Verify it's loaded
docker compose run --rm sora-extend env | grep OPENAI
```

### "Sora job failed"
- Sora might reject prompts with people/copyrighted content
- Check Sora API status: https://status.openai.com
- Verify Sora 2 access enabled on your account

### "ElevenLabs generation failed"
- Check API key is valid
- Verify voice ID exists: https://elevenlabs.io/app/voice-library
- Ensure sufficient credits in ElevenLabs account

### "Video compositing failed"
- Check FFmpeg is installed in container (it is by default)
- Ensure both B-roll and voiceover generated successfully
- Check disk space in output directory

## 📈 Production Tips

### Weekly Content Workflow
```bash
# Monday: Write 3-5 LinkedIn posts
# Tuesday: Generate all videos
for post in posts/*.txt; do
  name=$(basename "$post" .txt)
  docker compose run --rm sora-extend python src/linkedin_video_generator.py \
    --input "$post" \
    --name "$name"
done

# Wednesday-Sunday: Schedule posts
```

### A/B Testing
- Test different visual themes for same content
- Compare engagement: 30s vs 60s videos
- Track which topics resonate most

### Analytics
- Track video views, comments, shares
- Iterate on script style based on performance
- Refine theme selection based on engagement data

## 🚀 Next Steps

1. **Generate your first video** with the Steve Jobs example
2. **Review output quality** and iterate on prompts if needed
3. **Batch process** multiple LinkedIn posts
4. **Schedule regular posting** for consistent content
5. **Track performance** and optimize based on data

## 📧 Support

For issues or questions:
- Check this README first
- Review logs in `output/linkedin_videos/[video_name]/`
- Verify API keys and credits
- Test components individually (script generator → full pipeline)

---

**Ready to turn your LinkedIn posts into engaging videos?** Start with the Quick Start guide above!

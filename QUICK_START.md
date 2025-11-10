# LinkedIn-to-Video: Quick Start Guide

## ✅ What's Been Built

Your LinkedIn-to-video system is **ready to use**! Here's what's included:

### Core Components
- ✅ **Script Generator** (`src/script_generator.py`) - GPT-4o converts posts to speaking scripts
- ✅ **Video Orchestrator** (`src/linkedin_video_generator.py`) - End-to-end automation
- ✅ **Studio Templates** (`prompts/studio_templates.json`) - 4 visual themes
- ✅ **Voiceover System** (existing) - ElevenLabs integration
- ✅ **Docker Setup** - Container-ready deployment

### What Works
- ✅ Script generation tested with Steve Jobs example
- ✅ Theme auto-detection (classic/modern/intimate/energetic)
- ✅ Natural language optimization
- ✅ Parallel processing architecture
- ✅ Complete documentation

## 🚀 Generate Your First Video

### Step 1: Create a LinkedIn Post File

```bash
cd /home/adorosario/quick-and-dirty/sora-extend

cat > my_post.txt << 'EOF'
Steve Jobs on Quality

Most companies focus on shipping fast. Get it out there, iterate later.

But Jobs understood something deeper: first impressions are permanent.
You can't patch your way to excellence.

"Quality is more important than quantity. One home run is much better
than two doubles."

So ask yourself: are you shipping fast, or are you shipping right?
EOF
```

### Step 2: Generate the Video

```bash
docker compose run --rm sora-extend python src/linkedin_video_generator.py \
  --input my_post.txt \
  --name steve_jobs_quality \
  --duration 60 \
  --keep-intermediates
```

### Step 3: Wait ~5 Minutes

The system will:
1. Generate script from your post (10s)
2. Generate Sora studio B-roll in parallel with voiceover (3-5 min)
3. Composite final video (30s)

### Step 4: Get Your Video

```bash
# Final video location
ls -lh output/linkedin_videos/steve_jobs_quality/steve_jobs_quality.mp4

# All files
tree output/linkedin_videos/steve_jobs_quality/
├── steve_jobs_quality.mp4    ← FINAL VIDEO
├── script.json               ← Generated script
├── studio_broll.mp4          ← Sora footage
└── voiceover.mp3             ← Your voice
```

## 🧪 Test Script Generation First

Want to see the script before spending money on video generation?

```bash
docker compose run --rm sora-extend python src/script_generator.py \
  --input my_post.txt \
  --duration 60
```

**Output example**:
```
✓ Script generated successfully!
  Visual Theme: classic_studio
  Word Count: 156
  Estimated Duration: 60s

Script:
Hey there! Let's talk about a lesson from Steve Jobs that's as
relevant today as it ever was. You know, most companies these days
are all about speed...

Theme Rationale: The theme 'Classic Studio' fits perfectly as it
emphasizes leadership, quality, and principles—key elements of
Jobs' philosophy.
```

## 💰 Cost Estimate

**Per 60-second video**:
- GPT-4o script: $0.10
- Sora 2 B-roll: $3.00
- ElevenLabs VO: $0.20
- **Total: $3.30**

**Per 30-second video**:
- GPT-4o script: $0.10
- Sora 2 B-roll: $1.50
- ElevenLabs VO: $0.10
- **Total: $1.70**

## 🎨 Visual Themes (Auto-Selected)

Your system intelligently picks the right studio based on content:

### Classic Studio
**Triggers**: leadership, quality, wisdom, experience, principles
**Style**: Warm lighting, vintage mic, bookshelf, cozy

### Modern Studio
**Triggers**: tech, innovation, AI, startup, future, digital
**Style**: Blue LEDs, glass desk, minimalist, sharp

### Intimate Studio
**Triggers**: personal, story, journey, reflection, career
**Style**: Close-ups, window light, soft focus, warm

### Energetic Studio
**Triggers**: motivation, action, hustle, success, momentum
**Style**: Dynamic camera, high contrast, vibrant colors

## 📋 Your Configuration

Already set up in `.env`:
```bash
OPENAI_API_KEY=sk-proj-...  ✅
ELEVENLABS_API_KEY=sk_...   ✅
ELEVENLABS_VOICE_ID=XrEx... ✅
```

## 🎯 Next Steps

1. **Test with Steve Jobs post** (included in `test_posts/`)
2. **Review the generated video quality**
3. **Iterate on Sora prompts** if needed (edit `prompts/studio_templates.json`)
4. **Batch process** your weekly LinkedIn posts
5. **Track performance** and optimize

## 📖 Full Documentation

See `README_LINKEDIN_VIDEOS.md` for:
- Complete API reference
- Advanced usage patterns
- Troubleshooting guide
- Production workflows
- Cost optimization tips

## ⚡ Pro Tips

**Weekly Workflow**:
```bash
# Generate multiple videos at once
for post in posts/*.txt; do
  name=$(basename "$post" .txt)
  docker compose run --rm sora-extend python src/linkedin_video_generator.py \
    --input "$post" \
    --name "$name" \
    --duration 60
done
```

**Cost Savings**:
- Use 30s videos for higher engagement at lower cost
- Test script generation before full pipeline
- Batch similar posts to reuse visual themes

**Quality Tips**:
- Strong hook in first 5 seconds crucial
- Keep posts focused on 1-2 key ideas
- Test different themes for same content
- Review script before generating video

---

**You're all set!** Start with the Steve Jobs example, then create videos from your LinkedIn posts. 🚀

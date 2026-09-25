# DeepSentinel Deployment Master Guide (Option B)
## Global Access on Phones, Tablets, and Desktops via Vercel & Hugging Face Spaces

This guide walks you through deploying DeepSentinel so that anyone—including thesis panelists, advisors, and peer evaluators—can open the web app on their **iPhones, Android phones, iPads, tablets, or laptops** anywhere in the world and test **live video analysis with 100% of all real AI models**.

---

### Architecture Overview

```
                               GLOBAL DEPLOYMENT ARCHITECTURE
   📱 iPhone / Android / iPad
   💻 Mac / PC / Tablet
               │
               ▼
   🌐 Vercel Edge Network (`https://deepsentinel.vercel.app`)
      • Serves responsive HTML5/CSS3 frontend.
      • Touch-optimized video player and timeline scrubbing.
      • Real-time SSE telemetry animations.
               │
               ▼  (Proxied / Direct API Calls)
   🤗 Hugging Face Spaces GPU Container (`https://your-space.hf.space`)
      • Hardware: Nvidia T4 GPU (16GB VRAM) / 50GB SSD.
      • Models: Wav2Vec 2.0 + ViT-B/16 + Whisper + BERT + InsightFace.
      • Core Checkpoint: `best_phase2_adapted.pt` (299D Hybrid Bottleneck).
```

---

### Phase 1: Deploy Frontend to Vercel (2 Minutes)

1. Go to [Vercel](https://vercel.com) and log in with your GitHub account.
2. Click **"Add New..."** → **"Project"**.
3. Select your repository: `gjvlio/emotion-based-multimodal-deepfake-detector`.
4. Configure Project Settings:
   - **Framework Preset:** `Other`
   - **Root Directory:** `./`
   - **Build Command:** *(Leave empty)*
   - **Output Directory:** *(Leave empty)*
5. Click **"Deploy"**.
6. Within 30 seconds, Vercel will give you a public production URL:
   `https://deepsentinel.vercel.app` (or similar).

> [!NOTE]
> The repository already includes [`vercel.json`](file:///d:/Documents/Programming/Thesis_G10/vercel.json) in the root directory, which automatically maps routes (`/`, `/upload`, `/analyzing`, `/results`, `/benchmarks`, `/about/*`) to the responsive single-page application.

---

### Phase 2: Deploy Neural Backend to Hugging Face Spaces (3 Minutes)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and log in.
2. Click **"Create new Space"**:
   - **Space name:** `deepsentinel-engine` (or your preferred name).
   - **License:** `mit`.
   - **Select Space SDK:** **`Gradio`** (Selected by default) → Choose **`Blank`**.
     *(Note: Hugging Face charges for custom Docker, but Gradio is **100% FREE** with no credit card required!)*
   - **Space hardware:** Choose **`Free CPU`** (2 vCPU, 16 GB RAM) or **`T4 GPU`**.
   - **Space visibility:** **`Private`** (Protects your model weights and code from unauthorized access).
3. Click **"Create Space"**.
   ```bash
   # Clone the empty space repository
   git clone https://huggingface.co/spaces/YOUR_USERNAME/deepsentinel-engine temp_hf_space
   
   # Copy deployment files into the space directory
   cp deploy/hf_space/* temp_hf_space/
   cp -r src temp_hf_space/
   cp -r webapp temp_hf_space/
   cp -r checkpoints temp_hf_space/
   
   # Commit and push to Hugging Face
   cd temp_hf_space
   git add .
   git commit -m "feat: deploy DeepSentinel neural backend"
   git push origin main
   ```
4. Hugging Face will build the Docker container and start the service at:
   `https://YOUR_USERNAME-deepsentinel-engine.hf.space`

---

### Phase 3: Connect Vercel to Your Hugging Face Backend

You have two simple ways to connect the Vercel frontend to your live Hugging Face backend:

#### Method A: Automatic Vercel Proxy (Recommended)
Edit [`vercel.json`](file:///d:/Documents/Programming/Thesis_G10/vercel.json) in your repository and update the rewrites to point to your Hugging Face Space:

```json
{
  "version": 2,
  "rewrites": [
    {
      "source": "/detect/(.*)",
      "destination": "https://YOUR_USERNAME-deepsentinel-engine.hf.space/detect/$1"
    },
    {
      "source": "/detect",
      "destination": "https://YOUR_USERNAME-deepsentinel-engine.hf.space/detect"
    },
    {
      "source": "/warmup/status",
      "destination": "https://YOUR_USERNAME-deepsentinel-engine.hf.space/warmup/status"
    },
    {
      "source": "/health",
      "destination": "https://YOUR_USERNAME-deepsentinel-engine.hf.space/health"
    },
    {
      "source": "/(.*)",
      "destination": "/webapp/static/$1"
    }
  ]
}
```
Commit and push. Vercel will automatically redeploy in 15 seconds.

#### Method B: Browser Endpoint Setting (Instant)
In any browser (including mobile Safari or Chrome), open the developer console (or set via URL query):
```javascript
localStorage.setItem("ds_api_base", "https://YOUR_USERNAME-deepsentinel-engine.hf.space");
```
Our updated [`webapp/static/js/app.js`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/js/app.js) automatically reads `ds_api_base` and routes all API requests directly to your cloud GPU!

---

### Mobile & Tablet Testing Checklist

Once deployed, test on your mobile devices:

- [ ] **iPhone / iOS Safari:**
  - Open `https://deepsentinel.vercel.app`.
  - Notice the notch safe-area padding and black-translucent status bar.
  - Tap "Upload Video" → Choose **"Take Video"** with your iPhone camera or select from your Photo Library.
  - Scrub the timeline handles with your thumb: touch hitboxes are expanded to 44px for smooth drag-and-drop.
  - Tap **"Run Analysis"** and verify live step-by-step telemetry streaming.
- [ ] **iPad / Android Tablet:**
  - Test in both Portrait and Landscape orientations.
  - Verify that the video player expands cleanly to 45% viewport height and the 3-column telemetry bar scales smoothly.
- [ ] **Android Phone (Chrome / Firefox):**
  - Verify tap-highlight elimination (`-webkit-tap-highlight-color: transparent`).
  - Verify dominant emotion highlight banner wrapping and 6-emotion $\Delta$ breakdown cards.

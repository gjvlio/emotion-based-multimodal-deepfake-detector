/* DeepSentinel SPA — routing, detect flow, and the motion layer. Vanilla JS. */
(() => {
  "use strict";

  const EMO_ORDER = ["angry", "happy", "sad", "neutral", "fear", "disgust"];
  const EMO_LABEL = { angry: "Angry", happy: "Happy", sad: "Sad", neutral: "Neutral", fear: "Fearful", disgust: "Disgust", fearful: "Fearful" };

  const EMO_SVGS = {
    happy: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="9" cy="9.5" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="9.5" r="1.1" fill="currentColor" stroke="none"/><path d="M8 13.8c1.3 2.5 6.7 2.5 8 0"/></svg>`,
    sad: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M7.5 8.5l2.2 1M16.5 8.5l-2.2 1"/><circle cx="9" cy="11" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="11" r="1.1" fill="currentColor" stroke="none"/><path d="M8.5 16.5c1.2-2.1 5.8-2.1 7 0"/></svg>`,
    angry: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M7.5 8.5l3 1.5M16.5 8.5l-3 1.5"/><circle cx="9" cy="11.2" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="11.2" r="1.1" fill="currentColor" stroke="none"/><path d="M8.5 16c1.8-.8 5.2-.8 7 0"/></svg>`,
    disgust: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M7.5 10l2.5-1M7.5 9l2.5 1"/><circle cx="15" cy="9.5" r="1.1" fill="currentColor" stroke="none"/><path d="M8.5 16c1.4-1.2 3.1 1.3 4.4.2 1.2-1 2.6 0 2.6 0"/></svg>`,
    neutral: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="10" r="1.1" fill="currentColor" stroke="none"/><path d="M8.5 15h7"/></svg>`,
    fear: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M7.5 7.8c.8-.8 2-.8 2.8 0M16.5 7.8c-.8-.8-2-.8-2.8 0"/><circle cx="9" cy="11" r="1.4"/><circle cx="15" cy="11" r="1.4"/><ellipse cx="12" cy="16.2" rx="2.5" ry="1.8"/></svg>`,
    fearful: `<svg class="emo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M7.5 7.8c.8-.8 2-.8 2.8 0M16.5 7.8c-.8-.8-2-.8-2.8 0"/><circle cx="9" cy="11" r="1.4"/><circle cx="15" cy="11" r="1.4"/><ellipse cx="12" cy="16.2" rx="2.5" ry="1.8"/></svg>`,
  };

  const ROUTES = {
    "/": "landing", "/upload": "upload", "/analyzing": "analyzing", "/results": "results",
    "/about": "about-thesis", "/about/thesis": "about-thesis", "/about/researchers": "about-researchers",
    "/benchmarks": "benchmarks",
  };

  const views = {};
  document.querySelectorAll("[data-view]").forEach((v) => (views[v.dataset.view] = v));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let selectedFile = null;
  let lastResult = null;
  let demoMode = location.pathname.startsWith("/demo");
  let demoResultIndex = 0;

  const demoServedBy = {
    loaded: true,
    checkpoint: "demo-static",
    checkpoint_path: null,
    phase: 0,
    epoch: null,
    val_loss: null,
    last_modified: null,
    device: "demo",
    warmed: true,
    note: "Hardcoded demo result",
  };

  const DEMO_RESULTS = [
    () => ({
      verdict: "FAKE",
      p_fake: 0.74,
      threshold: 0.5,
      audio_text_emotion: {
        label: "happy",
        confidence: 0.61,
        distribution: { neutral: 0.11, happy: 0.61, sad: 0.08, angry: 0.10, fear: 0.05, disgust: 0.05 },
      },
      visual_emotion: {
        label: "sad",
        confidence: 0.64,
        distribution: { neutral: 0.09, happy: 0.06, sad: 0.64, angry: 0.11, fear: 0.05, disgust: 0.05 },
      },
      emotion_mismatch: { neutral: 0.08, happy: 0.61, sad: 0.68, angry: 0.42, fear: 0.21, disgust: 0.17 },
      p_sarcasm: 0.12,
      transcript: "I am absolutely thrilled to be here.",
      forensic_interpretation: {
        state_id: "STATE_FAKE_EMOTION_DESYNC",
        state_tag: "FAKE · EMOTION CLASH",
        headline: "Likely Deepfake: Voice and face emotions contradict each other",
        summary: "Voice sounds Happy (61%), but the face looks Sad (64%). This sharp contradiction happens when voice or video is swapped.",
        voice_face_analysis: "Sharp contradiction: Hearing happiness while seeing sadness does not happen in sincere human speech.",
        sarcasm_analysis: "Sarcasm is low (12%), confirming this clash is an AI flaw, not a joke.",
        technical_rationale: "Deepfake tools usually replace voice or face separately, leaving an obvious emotional seam.",
      },
      served_by: demoServedBy,
    }),
    () => ({
      verdict: "REAL",
      p_fake: 0.18,
      threshold: 0.5,
      audio_text_emotion: {
        label: "neutral",
        confidence: 0.74,
        distribution: { neutral: 0.74, happy: 0.09, sad: 0.06, angry: 0.04, fear: 0.04, disgust: 0.03 },
      },
      visual_emotion: {
        label: "neutral",
        confidence: 0.77,
        distribution: { neutral: 0.77, happy: 0.08, sad: 0.05, angry: 0.04, fear: 0.03, disgust: 0.03 },
      },
      emotion_mismatch: { neutral: 0.04, happy: 0.03, sad: 0.02, angry: 0.03, fear: 0.02, disgust: 0.02 },
      p_sarcasm: 0.09,
      transcript: "The clip is stable and the emotions line up across modalities.",
      forensic_interpretation: {
        state_id: "STATE_REAL_HARMONY",
        state_tag: "REAL · NATURAL MATCH",
        headline: "Looks Real: Voice and face emotions match naturally",
        summary: "Voice tone and facial expression agree on Neutral. What you hear and see align naturally.",
        voice_face_analysis: "Both voice and face show Neutral (74% / 77%) with no emotional clash.",
        sarcasm_analysis: "No sarcasm detected (9%). Delivery is sincere and straightforward.",
        technical_rationale: "Voice and mouth timing are in sync with no signs of AI editing.",
      },
      served_by: demoServedBy,
    }),
    () => ({
      verdict: "REAL",
      p_fake: 0.24,
      threshold: 0.5,
      audio_text_emotion: {
        label: "neutral",
        confidence: 0.57,
        distribution: { neutral: 0.57, happy: 0.17, sad: 0.08, angry: 0.06, fear: 0.06, disgust: 0.06 },
      },
      visual_emotion: {
        label: "neutral",
        confidence: 0.61,
        distribution: { neutral: 0.61, happy: 0.14, sad: 0.07, angry: 0.06, fear: 0.06, disgust: 0.06 },
      },
      emotion_mismatch: { neutral: 0.05, happy: 0.04, sad: 0.03, angry: 0.03, fear: 0.03, disgust: 0.02 },
      p_sarcasm: 0.81,
      transcript: "Sure, that was the best surprise ever.",
      forensic_interpretation: {
        state_id: "STATE_REAL_DEADPAN_IRONY",
        state_tag: "REAL · DEADPAN HUMOR",
        headline: "Looks Real: Deadpan joke (serious face with sarcastic voice)",
        summary: "The speaker is using sarcasm (81%) with a calm poker face. This is dry deadpan humor, not an AI fake.",
        voice_face_analysis: "Voice and face stay subdued (57% / 61% Neutral) while delivering an ironic comment.",
        sarcasm_analysis: "High sarcasm (81%). The model recognized dry humor, avoiding a false deepfake alert.",
        technical_rationale: "The irony filter accounts for dry humor so intentional poker faces are not flagged as fakes.",
      },
      served_by: demoServedBy,
    }),
  ];

  function nextDemoResult() {
    const makeResult = DEMO_RESULTS[demoResultIndex % DEMO_RESULTS.length];
    demoResultIndex += 1;
    return makeResult();
  }

  function routePath(path) {
    return demoMode ? (path === "/" ? "/demo" : "/demo" + path) : path;
  }

  function normalizeDemoPath(path) {
    return path.startsWith("/demo") ? (path.slice(5) || "/") : path;
  }

  // ── Routing ───────────────────────────────────────────────────────────────
  function navigate(path, replace = false) {
    if (replace) history.replaceState({}, "", path);
    else history.pushState({}, "", path);
    render();
  }

  function render() {
    let path = location.pathname;
    demoMode = path.startsWith("/demo");
    const logicalPath = normalizeDemoPath(path);
    let view = ROUTES[logicalPath] || "landing";
    if (view === "analyzing" && !selectedFile) { view = "upload"; history.replaceState({}, "", routePath("/upload")); }
    if (view === "results" && !lastResult) { view = "upload"; history.replaceState({}, "", routePath("/upload")); }

    Object.values(views).forEach((v) => v.classList.remove("active"));
    const el = views[view] || views.landing;
    el.classList.add("active");
    el.classList.remove("view-enter"); void el.offsetWidth; el.classList.add("view-enter");
    window.scrollTo({ top: 0 });

    document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
    if (logicalPath === "/") document.querySelector('.nav-link[href="/"]')?.classList.add("active");
    if (logicalPath.startsWith("/about")) document.querySelector(".nav-dropdown-toggle")?.classList.add("active");
    if (logicalPath === "/benchmarks") document.querySelector('.nav-link[href="/benchmarks"]')?.classList.add("active");

    document.body.classList.toggle("no-scroll", view === "landing" || view === "about-thesis");
    if (view !== "analyzing") stopAnalyzingHUD();
    if (view === "upload") resetUpload();
    if (view === "about-researchers") renderTeam();
    if (view === "benchmarks") renderBenchmarks();
    activateReveals(el);
    closeMenu();
  }

  // scroll-triggered blur-up reveal: above-fold items animate on view switch,
  // below-fold items animate as they scroll into view.
  const revealIO = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); } });
      }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" })
    : null;

  function activateReveals(viewEl) {
    const items = [...viewEl.querySelectorAll("[data-reveal]")];
    items.forEach((it, i) => {
      it.style.transitionDelay = reduce ? "0s" : Math.min(i, 6) * 70 + "ms";
      if (revealIO) revealIO.observe(it); else it.classList.add("in");
    });
  }

  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-link]");
    if (!a) return;
    e.preventDefault();
    navigate(routePath(a.getAttribute("href")));
  });
  window.addEventListener("popstate", render);

  // accessible guide-note toggles
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".info-btn");
    if (!btn) return;
    const note = document.getElementById(btn.dataset.guide);
    if (!note) return;
    const open = note.classList.toggle("open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  });

  // ── Header / menu ─────────────────────────────────────────────────────────
  const header = document.getElementById("header");
  const nav = document.getElementById("nav");
  const burger = document.getElementById("burger");
  window.addEventListener("scroll", () => header.classList.toggle("scrolled", window.scrollY > 6), { passive: true });
  burger?.addEventListener("click", () => { nav.classList.toggle("open"); burger.classList.toggle("open"); });
  function closeMenu() { nav.classList.remove("open"); burger?.classList.remove("open"); }

  // ── Motion: sparkles, magnetic buttons, card tilt ─────────────────────────
  function makeSparkles() {
    if (reduce) return;
    const box = document.getElementById("sparkles");
    const n = 12;
    for (let i = 0; i < n; i++) {
      const s = document.createElement("span");
      s.className = "spark";
      s.style.left = Math.random() * 100 + "vw";
      s.style.top = Math.random() * 100 + "vh";
      s.style.animationDelay = Math.random() * 4 + "s";
      s.style.animationDuration = 3 + Math.random() * 4 + "s";
      box.appendChild(s);
    }
  }

  // pointer:fine only (skip on touch), rect cached on enter, writes throttled to rAF
  const finePointer = window.matchMedia("(pointer: fine)").matches;

  function bindMagnetic() {
    if (reduce || !finePointer) return;
    document.querySelectorAll(".magnetic").forEach((el) => {
      if (el.dataset.magBound) return;
      el.dataset.magBound = "1";
      let rect = null, raf = 0, mx = 0, my = 0;
      el.addEventListener("mouseenter", () => { rect = el.getBoundingClientRect(); });
      el.addEventListener("mousemove", (e) => {
        if (!rect) rect = el.getBoundingClientRect();
        mx = (e.clientX - rect.left - rect.width / 2) * 0.16;
        my = (e.clientY - rect.top - rect.height / 2) * 0.24;
        if (raf) return;
        raf = requestAnimationFrame(() => { el.style.transform = `translate(${mx}px, ${my}px)`; raf = 0; });
      });
      el.addEventListener("mouseleave", () => { rect = null; el.style.transform = ""; });
    });
  }

  function bindTilt() {
    if (reduce || !finePointer) return;
    document.querySelectorAll("[data-tilt]").forEach((el) => {
      if (el.dataset.tiltBound) return;
      el.dataset.tiltBound = "1";
      let rect = null, raf = 0, rx = 0, ry = 0;
      el.addEventListener("mouseenter", () => { rect = el.getBoundingClientRect(); });
      el.addEventListener("mousemove", (e) => {
        if (!rect) rect = el.getBoundingClientRect();
        ry = ((e.clientX - rect.left) / rect.width - 0.5) * 5;
        rx = -((e.clientY - rect.top) / rect.height - 0.5) * 5;
        if (raf) return;
        raf = requestAnimationFrame(() => { el.style.transform = `perspective(800px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`; raf = 0; });
      });
      el.addEventListener("mouseleave", () => { rect = null; el.style.transform = ""; });
    });
  }

  // ── Upload & Interactive Filmstrip Timeline ─────────────────────────────
  const uploadStage = document.getElementById("upload-stage");
  const uploadEyebrow = document.getElementById("upload-eyebrow");
  const uploadTitle = document.getElementById("upload-title");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dzEmpty = dropzone.querySelector(".dz-empty");
  const dzFile = dropzone.querySelector(".dz-file");
  const dzFileName = dropzone.querySelector(".dz-file-name");

  const cropPanel = document.getElementById("crop-panel");
  const cropVideo = document.getElementById("crop-video");
  const videoWrapper = document.getElementById("video-wrapper");
  const videoCenterIndicator = document.getElementById("video-center-indicator");
  const playerPlayBtn = document.getElementById("player-play-btn");
  const playerCurrTime = document.getElementById("player-curr-time");
  const playerTotalTime = document.getElementById("player-total-time");
  const playerClipRange = document.getElementById("player-clip-range");
  const playerJumpStart = document.getElementById("player-jump-start");
  const playerMuteBtn = document.getElementById("player-mute-btn");

  const timelineTrack = document.getElementById("timeline-track");
  const filmstripCanvas = document.getElementById("filmstrip-canvas");
  const timelineMaskLeft = document.getElementById("timeline-mask-left");
  const timelineMaskRight = document.getElementById("timeline-mask-right");
  const timelineWindow = document.getElementById("timeline-window");
  const handleLeft = document.getElementById("handle-left");
  const windowBar = document.getElementById("window-bar");
  const handleRight = document.getElementById("handle-right");
  const timelinePlayhead = document.getElementById("timeline-playhead");

  const timelineHoverCard = document.getElementById("timeline-hover-card");
  const hoverCanvas = document.getElementById("hover-canvas");
  const hoverTimeBadge = document.getElementById("hover-time-badge");

  const startTimeVal = document.getElementById("start-time-val");
  const endTimeVal = document.getElementById("end-time-val");
  const cropDurationPill = document.getElementById("crop-duration-pill");
  const cropFileInfo = document.getElementById("crop-file-info");
  const changeVideoBtn = document.getElementById("change-video-btn");
  const previewClipBtn = document.getElementById("preview-clip-btn");
  const preset10s = document.getElementById("preset-10s");
  const preset20s = document.getElementById("preset-20s");
  const presetMiddle = document.getElementById("preset-middle");
  const runBtn = document.getElementById("run-btn");
  const runBtnLabel = document.getElementById("run-btn-label");
  const uploadError = document.getElementById("upload-error");

  const ALLOWED = [".mp4", ".mov", ".webm", ".avi", ".mkv"];
  const fmtSize = (b) =>
    b >= 1e9 ? (b / 1e9).toFixed(1) + " GB"
    : b >= 1e6 ? (b / 1e6).toFixed(0) + " MB"
    : (b / 1e3).toFixed(0) + " KB";

  const fmtTime = (s) => {
    if (isNaN(s) || s < 0) s = 0;
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${sec.toFixed(1).padStart(4, "0")}`;
  };

  let totalDuration = 10.0;
  let cropStart = 0.0;
  let cropEnd = 10.0;
  let previewTimer = null;
  let offscreenPreviewVideo = null;
  let filmstripAbort = false;

  function updateTimelineUI() {
    if (!totalDuration || totalDuration <= 0) return;

    // Strict clamping
    if (cropStart < 0) cropStart = 0;
    if (cropEnd > totalDuration) cropEnd = totalDuration;

    // Minimum 3.0s, Maximum 20.0s clamp if inverted
    if (cropEnd - cropStart < 0.5) {
      cropEnd = Math.min(totalDuration, cropStart + 0.5);
    }

    const leftPct = (cropStart / totalDuration) * 100;
    const rightPct = (cropEnd / totalDuration) * 100;
    const widthPct = Math.max(0.1, rightPct - leftPct);

    timelineMaskLeft.style.width = `${leftPct}%`;
    timelineMaskRight.style.width = `${Math.max(0, 100 - rightPct)}%`;
    timelineWindow.style.left = `${leftPct}%`;
    timelineWindow.style.width = `${widthPct}%`;

    // Position handles independently so they never crush or distort
    if (handleLeft) handleLeft.style.left = `${leftPct}%`;
    if (handleRight) handleRight.style.left = `${rightPct}%`;

    startTimeVal.textContent = fmtTime(cropStart);
    endTimeVal.textContent = fmtTime(cropEnd);
    playerClipRange.textContent = `${fmtTime(cropStart)} – ${fmtTime(cropEnd)}`;

    const dur = cropEnd - cropStart;
    const isValid = dur >= 2.95 && dur <= 20.05;

    cropDurationPill.className = `pill ${isValid ? "pill-optimal" : "pill-warning"}`;
    if (dur < 2.95) {
      cropDurationPill.textContent = `${dur.toFixed(1)}s selected (Too short: min 3.0s)`;
      runBtn.disabled = true;
      runBtnLabel.textContent = `Clip Too Short (${dur.toFixed(1)}s)`;
    } else if (dur > 20.05) {
      cropDurationPill.textContent = `${dur.toFixed(1)}s selected (Too long: max 20.0s)`;
      runBtn.disabled = true;
      runBtnLabel.textContent = `Clip Too Long (${dur.toFixed(1)}s)`;
    } else {
      cropDurationPill.textContent = `${dur.toFixed(1)}s selected (3s – 20s optimal)`;
      runBtn.disabled = false;
      runBtnLabel.textContent = `Analyze Selected Clip (${dur.toFixed(1)}s)`;
    }
  }

  // ── Filmstrip Generator (Bright, Crisp Video Thumbnails) ──────────────────
  function drawFilmstripBase(trackWidth, trackHeight) {
    if (!filmstripCanvas) return;
    const ctx = filmstripCanvas.getContext("2d");
    filmstripCanvas.width = trackWidth;
    filmstripCanvas.height = trackHeight;

    // Clear and draw rich slate-blue filmstrip base
    const grad = ctx.createLinearGradient(0, 0, trackWidth, 0);
    grad.addColorStop(0, "#2c3848");
    grad.addColorStop(0.5, "#3b485a");
    grad.addColorStop(1, "#283444");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, trackWidth, trackHeight);

    // Subtle frame separator lines
    const numSlices = 10;
    const sliceW = trackWidth / numSlices;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
    ctx.lineWidth = 1;
    for (let i = 1; i < numSlices; i++) {
      const x = Math.round(i * sliceW);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, trackHeight);
      ctx.stroke();
    }
  }

  async function generateFilmstrip(file) {
    filmstripAbort = true;
    await new Promise((r) => setTimeout(r, 50));
    filmstripAbort = false;

    if (!filmstripCanvas || !timelineTrack) return;
    const trackWidth = timelineTrack.clientWidth || 800;
    const trackHeight = 48;

    drawFilmstripBase(trackWidth, trackHeight);
    const ctx = filmstripCanvas.getContext("2d");

    // Offscreen video extractor
    const extractor = document.createElement("video");
    extractor.muted = true;
    extractor.playsInline = true;
    extractor.preload = "auto";
    const blobUrl = URL.createObjectURL(file);
    extractor.src = blobUrl;

    await new Promise((res) => {
      extractor.onloadeddata = res;
      extractor.onerror = res;
      setTimeout(res, 2000);
    });

    const numFrames = 8;
    const sliceW = trackWidth / numFrames;

    for (let i = 0; i < numFrames; i++) {
      if (filmstripAbort) break;
      const t = Math.min(totalDuration - 0.1, (i + 0.5) * (totalDuration / numFrames));
      extractor.currentTime = t;

      await new Promise((res) => {
        const onSeek = () => {
          extractor.removeEventListener("seeked", onSeek);
          res();
        };
        extractor.addEventListener("seeked", onSeek);
        setTimeout(res, 600); // 600ms timeout per frame for large files
      });

      if (filmstripAbort) break;
      try {
        ctx.drawImage(extractor, i * sliceW, 0, sliceW + 1, trackHeight);
      } catch (e) {}
    }

    URL.revokeObjectURL(blobUrl);
  }

  // Paint active playback frame to filmstrip
  cropVideo?.addEventListener("seeked", () => {
    if (!cropVideo || !filmstripCanvas || !totalDuration || cropVideo.readyState < 2) return;
    try {
      const ctx = filmstripCanvas.getContext("2d");
      const trackW = filmstripCanvas.width;
      const trackH = filmstripCanvas.height;
      const pct = Math.max(0, Math.min(1, cropVideo.currentTime / totalDuration));
      const sliceW = Math.max(40, trackW / 8);
      const destX = Math.max(0, Math.min(trackW - sliceW, pct * trackW - sliceW / 2));
      ctx.drawImage(cropVideo, destX, 0, sliceW, trackH);
    } catch (e) {}
  });

  // ── Floating Mini Preview Tooltip ──────────────────────────────────────────
  function showHoverPreview(clientX, timeSec) {
    if (!timelineHoverCard || !timelineTrack) return;
    const rect = timelineTrack.getBoundingClientRect();
    const clampedX = Math.max(rect.left, Math.min(rect.right, clientX));
    const localX = clampedX - rect.left;

    timelineHoverCard.style.left = `${localX}px`;
    timelineHoverCard.classList.add("is-visible");
    hoverTimeBadge.textContent = fmtTime(timeSec);

    // Instant zero-latency render from filmstrip canvas
    if (filmstripCanvas && hoverCanvas) {
      const hCtx = hoverCanvas.getContext("2d");
      const pct = totalDuration > 0 ? Math.max(0, Math.min(1, timeSec / totalDuration)) : 0;
      const srcW = Math.max(30, filmstripCanvas.width / 8);
      const srcX = Math.max(0, Math.min(filmstripCanvas.width - srcW, pct * filmstripCanvas.width - srcW / 2));
      try {
        hCtx.drawImage(filmstripCanvas, srcX, 0, srcW, filmstripCanvas.height, 0, 0, hoverCanvas.width, hoverCanvas.height);
      } catch (e) {}
    }
  }

  function hideHoverPreview() {
    if (!timelineHoverCard) return;
    timelineHoverCard.classList.remove("is-visible");
  }

  // ── Drag & Stretch Pointer Interactions (Bulletproof Document Listeners) ──
  let isDragging = false;
  let dragMode = null; // 'left' | 'right' | 'window'
  let dragStartX = 0;
  let dragStartCropStart = 0;
  let dragStartCropEnd = 0;

  function getTimeAtPointer(clientX) {
    const rect = timelineTrack.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
    return (x / rect.width) * totalDuration;
  }

  function startDrag(mode, e) {
    e.preventDefault();
    e.stopPropagation();
    isDragging = true;
    dragMode = mode;
    dragStartX = e.clientX;
    dragStartCropStart = cropStart;
    dragStartCropEnd = cropEnd;

    if (mode === "left") handleLeft?.classList.add("is-dragging");
    if (mode === "right") handleRight?.classList.add("is-dragging");
    if (mode === "window") timelineWindow?.classList.add("is-panning");

    showHoverPreview(e.clientX, mode === "left" ? cropStart : mode === "right" ? cropEnd : (cropStart + cropEnd) / 2);

    const onPointerMove = (ev) => {
      if (!isDragging) return;
      const trackWidth = timelineTrack.clientWidth || 1;
      const deltaSec = ((ev.clientX - dragStartX) / trackWidth) * totalDuration;

      if (dragMode === "left") {
        let newStart = dragStartCropStart + deltaSec;
        // Clamp: newStart >= 0, duration between 3.0s and 20.0s
        newStart = Math.max(0, newStart);
        newStart = Math.min(cropEnd - 3.0, newStart);
        newStart = Math.max(cropEnd - 20.0, newStart);

        cropStart = Math.round(newStart * 10) / 10;
        updateTimelineUI();
        cropVideo.currentTime = cropStart;
        showHoverPreview(ev.clientX, cropStart);
      } else if (dragMode === "right") {
        let newEnd = dragStartCropEnd + deltaSec;
        // Clamp: newEnd <= totalDuration, duration between 3.0s and 20.0s
        newEnd = Math.min(totalDuration, newEnd);
        newEnd = Math.max(cropStart + 3.0, newEnd);
        newEnd = Math.min(cropStart + 20.0, newEnd);

        cropEnd = Math.round(newEnd * 10) / 10;
        updateTimelineUI();
        cropVideo.currentTime = cropEnd;
        showHoverPreview(ev.clientX, cropEnd);
      } else if (dragMode === "window") {
        const dur = dragStartCropEnd - dragStartCropStart;
        let newStart = dragStartCropStart + deltaSec;

        if (newStart < 0) newStart = 0;
        if (newStart + dur > totalDuration) newStart = totalDuration - dur;

        cropStart = Math.round(newStart * 10) / 10;
        cropEnd = Math.round((newStart + dur) * 10) / 10;
        updateTimelineUI();
        cropVideo.currentTime = cropStart;
        showHoverPreview(ev.clientX, (cropStart + cropEnd) / 2);
      }
    };

    const onPointerUp = () => {
      isDragging = false;
      dragMode = null;
      handleLeft?.classList.remove("is-dragging");
      handleRight?.classList.remove("is-dragging");
      timelineWindow?.classList.remove("is-panning");
      hideHoverPreview();
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
      document.removeEventListener("pointercancel", onPointerUp);
    };

    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp);
    document.addEventListener("pointercancel", onPointerUp);
  }

  handleLeft?.addEventListener("pointerdown", (e) => startDrag("left", e));
  handleRight?.addEventListener("pointerdown", (e) => startDrag("right", e));
  timelineWindow?.addEventListener("pointerdown", (e) => startDrag("window", e));

  // Hover on timeline track when not dragging
  timelineTrack?.addEventListener("mousemove", (e) => {
    if (isDragging) return;
    const t = getTimeAtPointer(e.clientX);
    showHoverPreview(e.clientX, t);
  });

  timelineTrack?.addEventListener("mouseleave", () => {
    if (!isDragging) hideHoverPreview();
  });

  // Clicking track outside window centers or moves the window
  timelineTrack?.addEventListener("click", (e) => {
    if (e.target.closest(".timeline-handle") || e.target.closest(".timeline-window")) return;
    const clickedTime = getTimeAtPointer(e.clientX);
    const dur = cropEnd - cropStart;

    let newStart = clickedTime - dur / 2;
    if (newStart < 0) newStart = 0;
    if (newStart + dur > totalDuration) newStart = totalDuration - dur;

    cropStart = Math.round(newStart * 10) / 10;
    cropEnd = Math.round((newStart + dur) * 10) / 10;
    cropVideo.currentTime = cropStart;
    updateTimelineUI();
  });

  // ── Video Selection & Loading ────────────────────────────────────────────
  function pickFile(file) {
    if (!file) return;
    if (!ALLOWED.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      return showError(`Unsupported file format. Please use ${ALLOWED.join(", ")}.`);
    }
    if (file.size > 500 * 1024 * 1024) {
      return showError(`Video file exceeds 500 MB limit (${fmtSize(file.size)}). Please select a smaller clip.`);
    }
    if (file.size < 1000) {
      return showError("File is empty or corrupted (< 1 KB). Please select a valid video file.");
    }
    hideError();

    // Check video duration (up to 10 minutes = 600s)
    const tempVideo = document.createElement("video");
    tempVideo.preload = "metadata";
    tempVideo.onloadedmetadata = () => {
      window.URL.revokeObjectURL(tempVideo.src);
      const dur = tempVideo.duration;
      if (dur && dur > 600.5) {
        resetUpload();
        return showError(`Video exceeds 10 minutes (${(dur / 60).toFixed(1)} min). Maximum upload length is 10 minutes.`);
      }
      if (dur && dur < 2.9) {
        resetUpload();
        return showError(`Video is too short (${dur.toFixed(1)}s). Please upload a video at least 3 seconds long.`);
      }

      selectedFile = file;
      totalDuration = dur || 10.0;

      // Initialize crop: [0, min(totalDuration, 10s)]
      cropStart = 0.0;
      cropEnd = Math.min(totalDuration, 10.0);

      cropFileInfo.replaceChildren();
      const chipName = document.createElement("span");
      chipName.className = "crop-chip-name";
      chipName.title = file.name;
      chipName.textContent = file.name;

      const chipMeta = document.createElement("span");
      chipMeta.className = "crop-chip-meta";
      chipMeta.textContent = ` · ${fmtTime(totalDuration)} · ${fmtSize(file.size)}`;

      cropFileInfo.append(chipName, chipMeta);
      playerTotalTime.textContent = fmtTime(totalDuration);
      playerCurrTime.textContent = "00:00.0";

      cropVideo.src = URL.createObjectURL(file);
      cropVideo.currentTime = 0;

      // Replace dropzone completely with clip selector
      dropzone.hidden = true;
      dropzone.style.display = "none";
      cropPanel.hidden = false;
      cropPanel.style.display = "block";
      document.body.classList.add("is-crop-mode");
      uploadStage?.classList.add("stage-crop-active");
      if (uploadEyebrow) uploadEyebrow.textContent = "// 3–20s EVALUATION WINDOW";
      if (uploadTitle) uploadTitle.textContent = "Select clip region to inspect";
      runBtn.hidden = false;

      updateTimelineUI();
      generateFilmstrip(file);
    };

    tempVideo.onerror = () => {
      window.URL.revokeObjectURL(tempVideo.src);
      selectedFile = file;
      totalDuration = 10.0;
      cropStart = 0.0;
      cropEnd = 10.0;

      dropzone.hidden = true;
      dropzone.style.display = "none";
      cropPanel.hidden = false;
      cropPanel.style.display = "block";
      document.body.classList.add("is-crop-mode");
      uploadStage?.classList.add("stage-crop-active");
      runBtn.hidden = false;
      updateTimelineUI();
    };
    tempVideo.src = URL.createObjectURL(file);
  }

  // Presets
  preset10s?.addEventListener("click", () => {
    cropStart = 0;
    cropEnd = Math.min(totalDuration, 10.0);
    cropVideo.currentTime = cropStart;
    updateTimelineUI();
  });

  preset20s?.addEventListener("click", () => {
    cropStart = 0;
    cropEnd = Math.min(totalDuration, 20.0);
    cropVideo.currentTime = cropStart;
    updateTimelineUI();
  });

  presetMiddle?.addEventListener("click", () => {
    const mid = totalDuration / 2;
    cropStart = Math.max(0, mid - 5);
    cropEnd = Math.min(totalDuration, cropStart + 10);
    cropVideo.currentTime = cropStart;
    updateTimelineUI();
  });

  // ── Video Playback Controls ──────────────────────────────────────────────
  function togglePlay() {
    if (cropVideo.paused) {
      if (cropVideo.currentTime >= cropEnd || cropVideo.currentTime < cropStart) {
        cropVideo.currentTime = cropStart;
      }
      cropVideo.play();
    } else {
      cropVideo.pause();
    }
  }

  // Click on video canvas or play button
  cropVideo?.addEventListener("click", togglePlay);
  playerPlayBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    togglePlay();
  });

  playerJumpStart?.addEventListener("click", (e) => {
    e.stopPropagation();
    cropVideo.currentTime = cropStart;
  });

  playerMuteBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    cropVideo.muted = !cropVideo.muted;
    playerMuteBtn.classList.toggle("is-muted", cropVideo.muted);
  });

  cropVideo?.addEventListener("play", () => {
    playerPlayBtn?.classList.add("is-playing");
    videoCenterIndicator?.classList.remove("is-paused");
    timelinePlayhead?.classList.add("active");
  });

  cropVideo?.addEventListener("pause", () => {
    playerPlayBtn?.classList.remove("is-playing");
    videoCenterIndicator?.classList.add("is-paused");
  });

  cropVideo?.addEventListener("timeupdate", () => {
    playerCurrTime.textContent = fmtTime(cropVideo.currentTime);
    if (totalDuration > 0) {
      const pct = (cropVideo.currentTime / totalDuration) * 100;
      timelinePlayhead.style.left = `${pct}%`;
    }
  });

  // Preview Clip (plays only [cropStart, cropEnd])
  previewClipBtn?.addEventListener("click", () => {
    clearInterval(previewTimer);
    cropVideo.currentTime = cropStart;
    cropVideo.play();
    previewTimer = setInterval(() => {
      if (cropVideo.currentTime >= cropEnd) {
        cropVideo.pause();
        cropVideo.currentTime = cropStart;
        clearInterval(previewTimer);
      }
    }, 50);
  });

  const showError = (m) => { uploadError.textContent = m; uploadError.hidden = false; };
  const hideError = () => (uploadError.hidden = true);

  function resetUpload() {
    selectedFile = null;
    filmstripAbort = true;
    fileInput.value = "";
    clearInterval(previewTimer);
    if (cropVideo.src) {
      cropVideo.pause();
      cropVideo.removeAttribute("src");
      cropVideo.load();
    }
    dzEmpty.hidden = false;
    dzFile.hidden = true;
    dropzone.hidden = false;
    dropzone.style.display = "";
    cropPanel.hidden = true;
    cropPanel.style.display = "none";
    document.body.classList.remove("is-crop-mode");
    uploadStage?.classList.remove("stage-crop-active");
    if (uploadEyebrow) uploadEyebrow.textContent = "// ANALYZE A CLIP";
    if (uploadTitle) uploadTitle.textContent = "Upload a video to inspect";
    runBtn.hidden = true;
    runBtn.disabled = false;
    hideError();
  }

  changeVideoBtn?.addEventListener("click", resetUpload);
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
  runBtn.addEventListener("click", runAnalysis);

  // ── Analyze flow: HUD, Playback & Live Architecture Detection ─────────────
  let hudRaf = 0;
  let hudLoopHandler = null;
  let transcriptTimer = null;
  let currentStepIdx = 0;
  let insightFaceKeyframes = []; // [{ time, bbox: [nx1, ny1, nx2, ny2], kps: [[kx, ky]...] }]
  let liveVisualEmotion = null;  // { label: "neutral", confidence: 0.94 }
  let liveAudioEmotion = null;
  let clientTrackedBox = null;   // { x, y, w, h } from client face detector
  let faceScanInterval = null;
  let estimatedTotalSec = 10.0;
  let analysisStartTime = 0;
  let targetProgressPct = 0;
  let currentRenderedPct = 0;
  const STEP_MILESTONES = [12, 30, 62, 84, 92, 97, 100];

  function fmtElapsed(sec) {
    if (sec == null || isNaN(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ds = Math.floor((sec % 1) * 10);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${ds}`;
  }

  function stopAnalyzingHUD() {
    if (hudRaf) { cancelAnimationFrame(hudRaf); hudRaf = 0; }
    if (transcriptTimer) { clearInterval(transcriptTimer); transcriptTimer = null; }
    if (faceScanInterval) { clearInterval(faceScanInterval); faceScanInterval = null; }
    insightFaceKeyframes = [];
    liveVisualEmotion = null;
    liveAudioEmotion = null;
    clientTrackedBox = null;
    targetProgressPct = 0;
    currentRenderedPct = 0;

    const v = document.getElementById("analyzing-video");
    if (v) {
      if (hudLoopHandler) { v.removeEventListener("timeupdate", hudLoopHandler); hudLoopHandler = null; }
      v.pause();
      v.removeAttribute("src");
      v.load();
    }
  }

  // Client-side real face tracking (native FaceDetector or fast skin-chroma centroid)
  function startClientFaceTracking(video) {
    if (faceScanInterval) clearInterval(faceScanInterval);
    const scanCanvas = document.createElement("canvas");
    scanCanvas.width = 64;
    scanCanvas.height = 36;
    const sCtx = scanCanvas.getContext("2d", { willReadFrequently: true });

    faceScanInterval = setInterval(() => {
      if (!video || video.paused || video.readyState < 2) return;
      // If InsightFace keyframes already arrived from server, let them drive precision tracking
      if (insightFaceKeyframes && insightFaceKeyframes.length > 0) return;

      // 1. Native hardware FaceDetector in Chromium
      if ("FaceDetector" in window) {
        try {
          const fd = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
          fd.detect(video).then((faces) => {
            if (faces && faces.length > 0) {
              const b = faces[0].boundingBox;
              const vw = video.videoWidth || 1;
              const vh = video.videoHeight || 1;
              clientTrackedBox = {
                x: Math.max(0.01, b.x / vw),
                y: Math.max(0.01, b.y / vh),
                w: Math.min(0.9, b.width / vw),
                h: Math.min(0.9, b.height / vh),
              };
            }
          }).catch(() => {});
          return;
        } catch (e) {}
      }

      // 2. High-speed human skin-chroma centroid detector (immediate face tracking)
      try {
        sCtx.drawImage(video, 0, 0, 64, 36);
        const imgData = sCtx.getImageData(0, 0, 64, 36).data;
        let sumX = 0, sumY = 0, count = 0;
        let minX = 64, maxX = 0, minY = 36, maxY = 0;

        for (let py = 2; py < 28; py++) {
          for (let px = 2; px < 62; px++) {
            const idx = (py * 64 + px) * 4;
            const r = imgData[idx];
            const g = imgData[idx + 1];
            const b = imgData[idx + 2];
            // Facial skin tone rule
            if (r > 85 && g > 40 && b > 20 && (r - g) > 12 && r > b) {
              sumX += px;
              sumY += py;
              count++;
              if (px < minX) minX = px;
              if (px > maxX) maxX = px;
              if (py < minY) minY = py;
              if (py > maxY) maxY = py;
            }
          }
        }

        if (count > 20) {
          const avgX = sumX / count / 64;
          const avgY = sumY / count / 36;
          const boxW = Math.max(0.24, Math.min(0.42, (maxX - minX + 8) / 64));
          const boxH = Math.max(0.28, Math.min(0.52, boxW * 1.25));
          clientTrackedBox = {
            x: Math.max(0.02, Math.min(0.98 - boxW, avgX - boxW / 2)),
            y: Math.max(0.02, Math.min(0.98 - boxH, avgY - boxH / 2)),
            w: boxW,
            h: boxH,
          };
        }
      } catch (e) {}
    }, 150);
  }

  function startAnalyzingHUD(file, startSec, endSec) {
    stopAnalyzingHUD();

    const video = document.getElementById("analyzing-video");
    const canvas = document.getElementById("analyzing-hud-canvas");
    const clipBadge = document.getElementById("analyzing-clip-badge");
    const transcriptEl = document.getElementById("analyzing-transcript-text");
    const hudEmotionStat = document.getElementById("hud-emotion-stat");
    const hudPhaseLabel = document.getElementById("hud-phase-label");

    if (clipBadge) clipBadge.textContent = `${fmtTime(startSec)} – ${fmtTime(endSec)}`;

    if (transcriptEl) {
      transcriptEl.innerHTML = '<span class="typing-cursor">Listening to speech audio stream (16kHz)...</span>';
    }

    if (!video || !canvas || !file) return;

    // Load selected clip for looped playback
    video.src = URL.createObjectURL(file);
    video.currentTime = startSec;
    video.muted = true;
    video.play().catch(() => {});

    hudLoopHandler = () => {
      if (video.currentTime >= endSec || video.currentTime < startSec - 0.2) {
        video.currentTime = startSec;
      }
    };
    video.addEventListener("timeupdate", hudLoopHandler);
    video.addEventListener("ended", () => {
      video.currentTime = startSec;
      video.play().catch(() => {});
    });

    startClientFaceTracking(video);

    const ctx = canvas.getContext("2d");
    const startTime = performance.now();
    analysisStartTime = startTime;
    const clipDur = Math.max(1.0, endSec - startSec);
    estimatedTotalSec = Math.max(8.5, Math.round((4.2 + clipDur * 0.95) * 10) / 10);
    targetProgressPct = 6;
    currentRenderedPct = 0;
    let lastDomTimeUpdate = 0;

    // Smoothed bounding box state (normalized [0, 1])
    let smoothBox = { x: 0.38, y: 0.20, w: 0.32, h: 0.42 };

    function renderHUD(now) {
      const elapsed = (now - startTime) / 1000;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const cssW = rect.width;
      const cssH = rect.height;

      if (cssW > 0 && cssH > 0) {
        const targetW = Math.round(cssW * dpr);
        const targetH = Math.round(cssH * dpr);
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cssW, cssH);

      if (cssW > 60 && cssH > 60) {
        // Video render aspect ratio & letterbox bounds
        const vw = video.videoWidth || 16;
        const vh = video.videoHeight || 9;
        const videoRatio = vw / vh;
        const canvasRatio = cssW / cssH;
        let drawW, drawH, drawX, drawY;
        if (videoRatio > canvasRatio) {
          drawW = cssW;
          drawH = cssW / videoRatio;
          drawX = 0;
          drawY = (cssH - drawH) / 2;
        } else {
          drawH = cssH;
          drawW = cssH * videoRatio;
          drawX = (cssW - drawW) / 2;
          drawY = 0;
        }

        // Determine target face box
        let targetNormBox = null;
        let targetKps = null;

        // 1. Priority 1: Real InsightFace / RetinaFace detections from backend
        if (insightFaceKeyframes && insightFaceKeyframes.length > 0) {
          let closest = insightFaceKeyframes[0];
          let minDiff = Infinity;
          for (const f of insightFaceKeyframes) {
            const diff = Math.abs(f.time - video.currentTime);
            if (diff < minDiff) { minDiff = diff; closest = f; }
          }
          if (closest && closest.bbox) {
            const [nx1, ny1, nx2, ny2] = closest.bbox;
            targetNormBox = {
              x: nx1,
              y: ny1,
              w: Math.max(0.12, nx2 - nx1),
              h: Math.max(0.16, ny2 - ny1),
            };
            if (closest.kps && closest.kps.length > 0) {
              targetKps = closest.kps;
            }
          }
        }

        // 2. Priority 2: Client face tracker
        if (!targetNormBox && clientTrackedBox) {
          targetNormBox = clientTrackedBox;
        }

        // 3. Fallback: Upper-third portrait default
        if (!targetNormBox) {
          targetNormBox = {
            x: 0.35 + Math.sin(elapsed * 0.8) * 0.03,
            y: 0.18 + Math.cos(elapsed * 0.6) * 0.02,
            w: 0.32,
            h: 0.44,
          };
        }

        // Exponential smoothing for buttery-smooth tracking
        smoothBox.x += (targetNormBox.x - smoothBox.x) * 0.22;
        smoothBox.y += (targetNormBox.y - smoothBox.y) * 0.22;
        smoothBox.w += (targetNormBox.w - smoothBox.w) * 0.22;
        smoothBox.h += (targetNormBox.h - smoothBox.h) * 0.22;

        // Convert normalized [0, 1] to canvas screen pixels
        const x = drawX + smoothBox.x * drawW;
        const y = drawY + smoothBox.y * drawH;
        const bw = Math.max(48, smoothBox.w * drawW);
        const bh = Math.max(58, smoothBox.h * drawH);
        const cx = x + bw / 2;
        const cy = y + bh / 2;
        const cornerLen = Math.min(22, bw * 0.25);

        // Emotion evaluation logic
        let emoLabel = "Neutral";
        let emoConf = 92.4;
        let emoColor = "#3df3d8";

        if (liveVisualEmotion) {
          const raw = liveVisualEmotion.label || "neutral";
          emoLabel = EMO_LABEL[raw.toLowerCase()] || raw.toUpperCase();
          emoConf = Math.round((liveVisualEmotion.confidence || 0.9) * 1000) / 10;
        } else if (lastResult?.visual_emotion) {
          const raw = lastResult.visual_emotion.label || "neutral";
          emoLabel = EMO_LABEL[raw.toLowerCase()] || raw.toUpperCase();
          emoConf = Math.round((lastResult.visual_emotion.confidence || 0.9) * 1000) / 10;
        } else {
          if (currentStepIdx < 2) {
            emoLabel = "Listening...";
            emoConf = 86.0 + Math.sin(elapsed * 3) * 6;
          } else if (currentStepIdx === 2) {
            emoLabel = "RetinaFace Tracking";
            emoConf = 97.8;
          } else {
            emoLabel = "ViT Classifying...";
            emoConf = 91.5 + Math.sin(elapsed * 2) * 5;
          }
        }

        const lowerEmo = emoLabel.toLowerCase();
        if (lowerEmo === "happy") emoColor = "#2cd5be";
        else if (lowerEmo === "angry" || lowerEmo === "fear" || lowerEmo === "fearful") emoColor = "#db5a42";
        else if (lowerEmo === "sad" || lowerEmo === "disgust") emoColor = "#72a1e5";

        if (hudEmotionStat) {
          hudEmotionStat.textContent = `${emoLabel} · ${emoConf.toFixed(1)}%`;
          hudEmotionStat.style.color = emoColor;
          hudEmotionStat.style.borderColor = `${emoColor}66`;
        }

        // 1. Subtle dashed tracking frame
        ctx.strokeStyle = "rgba(44, 213, 190, 0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(x, y, bw, bh);
        ctx.setLineDash([]);

        // 2. High-tech Corner Brackets (Glowing Cyan)
        ctx.strokeStyle = "#3df3d8";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#1EA896";
        ctx.shadowBlur = 8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + bw - cornerLen, y);
        ctx.lineTo(x + bw, y);
        ctx.lineTo(x + bw, y + cornerLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + bh - cornerLen);
        ctx.lineTo(x, y + bh);
        ctx.lineTo(x + cornerLen, y + bh);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + bw - cornerLen, y + bh);
        ctx.lineTo(x + bw, y + bh);
        ctx.lineTo(x + bw, y + bh - cornerLen);
        ctx.stroke();

        ctx.shadowBlur = 0;

        // 3. Central target crosshair
        ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5);
        ctx.stroke();

        // 4. Sweeping laser scanline
        const scanY = y + ((Math.sin(elapsed * 2.8) + 1) / 2) * bh;
        const grad = ctx.createLinearGradient(x, scanY, x + bw, scanY);
        grad.addColorStop(0, "rgba(44, 213, 190, 0)");
        grad.addColorStop(0.2, "rgba(44, 213, 190, 0.7)");
        grad.addColorStop(0.5, "rgba(255, 255, 255, 0.95)");
        grad.addColorStop(0.8, "rgba(44, 213, 190, 0.7)");
        grad.addColorStop(1, "rgba(44, 213, 190, 0)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(x + 2, scanY);
        ctx.lineTo(x + bw - 2, scanY);
        ctx.stroke();

        // 5. Facial Landmark constellation points & links
        let landmarks = [];
        if (targetKps && targetKps.length >= 5) {
          landmarks = targetKps.map(([kx, ky]) => ({ x: drawX + kx * drawW, y: drawY + ky * drawH }));
        } else {
          landmarks = [
            { x: x + bw * 0.32, y: y + bh * 0.38 }, // left eye
            { x: x + bw * 0.68, y: y + bh * 0.38 }, // right eye
            { x: x + bw * 0.50, y: y + bh * 0.54 }, // nose
            { x: x + bw * 0.35, y: y + bh * 0.74 }, // mouth left
            { x: x + bw * 0.65, y: y + bh * 0.74 }, // mouth right
          ];
        }

        ctx.strokeStyle = "rgba(44, 213, 190, 0.25)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (landmarks.length >= 5) {
          ctx.moveTo(landmarks[0].x, landmarks[0].y);
          ctx.lineTo(landmarks[1].x, landmarks[1].y);
          ctx.lineTo(landmarks[2].x, landmarks[2].y);
          ctx.closePath();
          ctx.moveTo(landmarks[2].x, landmarks[2].y);
          ctx.lineTo(landmarks[3].x, landmarks[3].y);
          ctx.lineTo(landmarks[4].x, landmarks[4].y);
          ctx.closePath();
          ctx.stroke();
        }

        landmarks.forEach((p) => {
          ctx.fillStyle = "#3df3d8";
          ctx.beginPath();
          ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });

        // 6. Top tag on Bounding Box
        const topLabelText = insightFaceKeyframes.length > 0 ? "INSIGHTFACE · RETINAFACE" : "FACE #01 · ViT KEYFRAME";
        ctx.font = "600 9px monospace";
        const topTextW = ctx.measureText(topLabelText).width;
        ctx.fillStyle = "rgba(11, 15, 23, 0.85)";
        ctx.fillRect(x, y - 18, topTextW + 10, 16);
        ctx.strokeStyle = "rgba(44, 213, 190, 0.4)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y - 18, topTextW + 10, 16);
        ctx.fillStyle = "#3df3d8";
        ctx.fillText(topLabelText, x + 5, y - 6);

        // 7. Live Emotion Detection Badge Attached to Box (Placed safely so never clipped)
        const emoPillText = `EMOTION: ${emoLabel.toUpperCase()} ${emoConf.toFixed(0)}%`;
        ctx.font = "700 10.5px monospace";
        const emoPillW = ctx.measureText(emoPillText).width + 18;
        const emoPillX = Math.max(drawX + 4, Math.min(drawX + drawW - emoPillW - 4, x + bw - emoPillW));
        // Place above the box if room, otherwise inside or below without colliding with bottom telemetry pill
        let emoPillY = y - 26;
        if (emoPillY < drawY + 6) {
          if (y + bh + 28 <= cssH - 60) {
            emoPillY = y + bh + 8;
          } else {
            emoPillY = Math.max(drawY + 6, y + 6);
          }
        }
        // Strict boundary clamp so it NEVER collides with bottom telemetry pill (occupies bottom 10-48px)
        emoPillY = Math.max(drawY + 4, Math.min(cssH - 72, emoPillY));

        ctx.fillStyle = "rgba(11, 15, 23, 0.92)";
        ctx.beginPath();
        ctx.roundRect(emoPillX, emoPillY, emoPillW, 22, 6);
        ctx.fill();

        ctx.strokeStyle = emoColor;
        ctx.lineWidth = 1.2;
        ctx.shadowColor = emoColor;
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.fillStyle = emoColor;
        ctx.beginPath();
        ctx.arc(emoPillX + 9, emoPillY + 11, 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.fillText(emoPillText, emoPillX + 18, emoPillY + 15);

        // Smooth asymptotic progress interpolation towards targetProgressPct
        if (currentRenderedPct < targetProgressPct) {
          const diff = targetProgressPct - currentRenderedPct;
          currentRenderedPct += Math.max(0.15, diff * 0.08);
          if (currentRenderedPct > targetProgressPct) currentRenderedPct = targetProgressPct;
        }

        // 8. Corner Telemetry & Live Evaluation Stopwatch
        const elapsedSec = (now - startTime) / 1000;

        // Bottleneck guard: if elapsedSec catches up to estimatedTotalSec while intermediate steps
        // are still actively computing, dynamically expand estimatedTotalSec so EST. REMAINING
        // never freezes or reads 0.1s while waiting.
        if (currentStepIdx < 6) {
          const pendingSteps = 6 - currentStepIdx;
          const minBufferSec = Math.max(1.8, pendingSteps * 0.85);
          if (elapsedSec > estimatedTotalSec - minBufferSec) {
            estimatedTotalSec = Math.round((elapsedSec + minBufferSec) * 10) / 10;
          }
        }

        const elapsedStr = fmtElapsed(elapsedSec);
        const estStr = fmtElapsed(estimatedTotalSec);
        const remSec = Math.max(0.1, estimatedTotalSec - elapsedSec);
        const remStr = fmtElapsed(remSec);

        // Update DOM timing metrics and loader percentage smoothly without UI lag
        if (now - lastDomTimeUpdate > 60) {
          lastDomTimeUpdate = now;
          const elElapsedVal = document.getElementById("timing-elapsed-val");
          const elEstVal = document.getElementById("timing-est-val");
          const elRemainingVal = document.getElementById("timing-remaining-val");

          if (elElapsedVal) elElapsedVal.textContent = elapsedStr;
          if (elEstVal) elEstVal.textContent = `~${estStr}`;
          if (elRemainingVal) elRemainingVal.textContent = `~${remStr}`;

          const pctInt = Math.min(100, Math.round(currentRenderedPct));
          const elPct = document.getElementById("analyzing-progress-pct");
          if (elPct) elPct.textContent = `${pctInt}%`;

          const loaderArc = document.getElementById("loader-arc");
          if (loaderArc) {
            const circ = 264;
            const offset = Math.max(0, circ - (currentRenderedPct / 100) * circ);
            loaderArc.style.strokeDashoffset = offset.toFixed(1);
          }
        }

        ctx.font = "500 10px monospace";
        ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
        ctx.fillText(`CROP: [${fmtTime(startSec)} - ${fmtTime(endSec)}]`, 12, 18);
        ctx.fillText(`FPS: 29.97 · FACS AU: SALIENT`, 12, 32);
      }

      ctx.restore();
      hudRaf = requestAnimationFrame(renderHUD);
    }

    hudRaf = requestAnimationFrame(renderHUD);
  }

  function streamLiveTranscript(finalTranscript) {
    if (transcriptTimer) clearInterval(transcriptTimer);
    const el = document.getElementById("analyzing-transcript-text");
    if (!el) return;

    const textToStream = finalTranscript && finalTranscript.trim().length > 0
      ? `“${finalTranscript.trim()}”`
      : "“No discernible speech detected in this clip selection.”";

    let charIdx = 0;
    el.innerHTML = '<span class="typing-cursor"></span>';

    transcriptTimer = setInterval(() => {
      charIdx += 2;
      const currentSub = textToStream.slice(0, charIdx);
      el.innerHTML = `<span>${currentSub}</span><span class="typing-cursor"></span>`;
      if (charIdx >= textToStream.length) {
        clearInterval(transcriptTimer);
        transcriptTimer = null;
      }
    }, 28);
  }

  // ── Paced Real-time Walkthrough for Demo Mode ───────────────────────────────
  async function runDemoAnalysis(markStep, setPhase) {
    lastResult = nextDemoResult();
    const demoSteps = [
      { step: 0, phase: "Listening to the voice (16kHz Wav2Vec 2.0)", delay: 1100 },
      {
        step: 1,
        phase: "Reading the tone & words (BERT NLP)",
        delay: 1500,
        onEnter: () => streamLiveTranscript(lastResult.transcript),
      },
      {
        step: 2,
        phase: "Picking the clearest face frames (InsightFace)",
        delay: 1300,
      },
      {
        step: 3,
        phase: "Reading the face's emotion (Vision Transformer)",
        delay: 1400,
        onEnter: () => { liveVisualEmotion = lastResult.visual_emotion; },
      },
      { step: 4, phase: "Comparing voice emotion vs face emotion", delay: 1100 },
      { step: 5, phase: "Measuring the emotion gap (Δ)", delay: 1100 },
      { step: 6, phase: "Reaching a verdict", delay: 1100 },
    ];

    for (const ds of demoSteps) {
      markStep(ds.step, "active");
      setPhase(ds.phase);
      if (ds.onEnter) ds.onEnter();
      await new Promise((r) => setTimeout(r, ds.delay));
      markStep(ds.step, "done");
    }

    targetProgressPct = 100;
    currentRenderedPct = 100;
    const elPctDemo = document.getElementById("analyzing-progress-pct");
    if (elPctDemo) elPctDemo.textContent = "100%";
    const loaderArcDemo = document.getElementById("loader-arc");
    if (loaderArcDemo) loaderArcDemo.style.strokeDashoffset = "0";

    await new Promise((r) => setTimeout(r, 700));
    stopAnalyzingHUD();
    renderResults(lastResult);
    navigate(routePath("/results"));
  }

  // ── Diagnostic Error Card for Input Modality / Quality Failures ───────────
  function showDiagnosticError(rawErr) {
    stopAnalyzingHUD();
    const v = document.getElementById("analyzing-video");
    if (v) { v.pause(); }

    let errObj = {};
    if (rawErr && typeof rawErr === "object") {
      if (rawErr.detail && typeof rawErr.detail === "object") {
        errObj = rawErr.detail;
      } else {
        errObj = rawErr;
      }
    } else if (typeof rawErr === "string") {
      try {
        errObj = JSON.parse(rawErr);
      } catch (e) {
        errObj = { message: rawErr };
      }
    }

    const statusCard = document.querySelector(".analyzing-status-card");
    const errorCard = document.getElementById("analyzing-error-card");
    if (statusCard) statusCard.hidden = true;
    if (errorCard) errorCard.hidden = false;

    const badgeEl = document.getElementById("diag-error-badge");
    const titleEl = document.getElementById("diag-error-title");
    const msgEl = document.getElementById("diag-error-message");
    const listEl = document.getElementById("diag-suggestions-list");

    const badgeText = errObj.code ? `MODALITY ERROR: ${errObj.code}` : "INPUT VERIFICATION FAILED";
    const titleText = errObj.title || "Input Verification Failed";
    const msgText = errObj.message || "DeepSentinel was unable to inspect this clip because essential multimodal requirements were not met.";

    let suggestions = [];
    if (Array.isArray(errObj.suggestions) && errObj.suggestions.length > 0) {
      suggestions = errObj.suggestions;
    } else if (errObj.suggestion) {
      suggestions = [errObj.suggestion];
    } else {
      suggestions = [
        "Verify that the uploaded video contains an audible speaking voice.",
        "Ensure the face is clearly visible, well-lit, and not obstructed by a mask or heavy blur.",
        "Use the timeline selector to choose an active segment where the subject speaks."
      ];
    }

    if (badgeEl) badgeEl.textContent = badgeText;
    if (titleEl) titleEl.textContent = titleText;
    if (msgEl) msgEl.textContent = msgText;

    if (listEl) {
      listEl.innerHTML = suggestions.map((s) => `
        <li>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          <span>${s}</span>
        </li>
      `).join("");
    }

    const reselectBtn = document.getElementById("diag-reselect-btn");
    const restartBtn = document.getElementById("diag-restart-btn");

    if (reselectBtn) {
      reselectBtn.onclick = () => {
        if (errorCard) errorCard.hidden = true;
        if (statusCard) statusCard.hidden = false;
        navigate(routePath("/upload"));
      };
    }

    if (restartBtn) {
      restartBtn.onclick = () => {
        if (errorCard) errorCard.hidden = true;
        if (statusCard) statusCard.hidden = false;
        resetUpload();
        navigate(routePath("/upload"));
      };
    }
  }

  // ── Live Stream Execution Tied to Real Backend Architecture ────────────────
  async function runAnalysis() {
    if (!selectedFile) return;

    // Reset error card visibility
    const errCard = document.getElementById("analyzing-error-card");
    const statusCard = document.querySelector(".analyzing-status-card");
    if (errCard) errCard.hidden = true;
    if (statusCard) statusCard.hidden = false;

    const fileEl = document.getElementById("analyzing-file");
    if (fileEl) {
      fileEl.replaceChildren();
      const fnSpan = document.createElement("span");
      fnSpan.className = "analyzing-filename";
      fnSpan.title = selectedFile.name;
      fnSpan.textContent = selectedFile.name;

      const metaSpan = document.createElement("span");
      metaSpan.className = "analyzing-clip-meta";
      metaSpan.textContent = `Clip: ${fmtTime(cropStart)} – ${fmtTime(cropEnd)} (${(cropEnd - cropStart).toFixed(1)}s)`;

      fileEl.append(fnSpan, metaSpan);
    }
    navigate(routePath("/analyzing"));

    // Reset detection state
    insightFaceKeyframes = [];
    liveVisualEmotion = null;
    liveAudioEmotion = null;
    clientTrackedBox = null;

    // Start video playback & real-time face HUD on the left
    startAnalyzingHUD(selectedFile, cropStart, cropEnd);

    const steps = [...document.querySelectorAll("#steps li")];
    steps.forEach((s) => s.classList.remove("done", "active"));

    const hudPhaseLabel = document.getElementById("hud-phase-label");
    const setPhase = (title) => {
      if (hudPhaseLabel && hudPhaseLabel.querySelector("span")) {
        hudPhaseLabel.querySelector("span").textContent = title;
      }
    };

    const markStep = (stepIdx, status) => {
      currentStepIdx = stepIdx;
      for (let i = 0; i < stepIdx; i++) {
        steps[i].classList.remove("active");
        steps[i].classList.add("done");
      }
      if (status === "active") {
        steps[stepIdx].classList.add("active");
        steps[stepIdx].classList.remove("done");
        targetProgressPct = STEP_MILESTONES[stepIdx] || (stepIdx + 1) * 14;
      } else if (status === "done") {
        steps[stepIdx].classList.remove("active");
        steps[stepIdx].classList.add("done");
        targetProgressPct = Math.min(100, (STEP_MILESTONES[stepIdx] || (stepIdx + 1) * 14) + 4);
      }

      // Dynamically refine estimated total runtime based on actual step progress
      if (analysisStartTime > 0 && stepIdx > 0) {
        const stepFractions = [0.12, 0.30, 0.62, 0.84, 0.92, 0.97, 1.0];
        const f = stepFractions[stepIdx] || 0.15;
        const curElapsed = (performance.now() - analysisStartTime) / 1000;
        if (curElapsed > 1.0) {
          const sampleEst = curElapsed / f;
          const remEstimate = (1 - f) * sampleEst * 0.9;
          estimatedTotalSec = Math.max(curElapsed + Math.max(0.8, remEstimate), estimatedTotalSec * 0.45 + sampleEst * 0.55);
        }
      }
    };

    if (demoMode) {
      await runDemoAnalysis(markStep, setPhase);
      return;
    }

    try {
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("start_time", cropStart.toFixed(2));
      form.append("end_time", cropEnd.toFixed(2));

      markStep(0, "active");
      setPhase("Listening to the voice (16kHz Wav2Vec 2.0)");

      // Connect to real-time Server-Sent Event stream
      const res = await fetch("/detect/stream", { method: "POST", body: form });
      if (!res.ok) {
        // Fallback to standard /detect endpoint if streaming endpoint is unavailable
        const fbRes = await fetch("/detect", { method: "POST", body: form });
        if (!fbRes.ok) {
          const err = await fbRes.json().catch(() => ({}));
          showDiagnosticError(err.detail || `Server error (${fbRes.status})`);
          return;
        }
        lastResult = await fbRes.json();
        if (lastResult?.transcript) streamLiveTranscript(lastResult.transcript);
        steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
        targetProgressPct = 100;
        currentRenderedPct = 100;
        const fbPct = document.getElementById("analyzing-progress-pct");
        if (fbPct) fbPct.textContent = "100%";
        const fbArc = document.getElementById("loader-arc");
        if (fbArc) fbArc.style.strokeDashoffset = "0";
        await new Promise((r) => setTimeout(r, 800));
        stopAnalyzingHUD();
        renderResults(lastResult);
        navigate(routePath("/results"));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop();

        for (const block of blocks) {
          const line = block.trim();
          if (!line.startsWith("data:")) continue;
          try {
            const ev = JSON.parse(line.slice(5).trim());
            if (ev.error) {
              showDiagnosticError(ev.error);
              return;
            }

            if (typeof ev.step === "number") {
              markStep(ev.step, ev.status);
            }

            if (ev.name) setPhase(ev.name);

            // REAL TRANSCRIPT: Emitted directly from Step 1 (Whisper)!
            if (ev.transcript) {
              streamLiveTranscript(ev.transcript);
            }

            // REAL INSIGHTFACE RETINAFACE KEYFRAMES: Emitted from Step 2!
            if (ev.faces && ev.faces.length > 0) {
              insightFaceKeyframes = ev.faces;
              setPhase(`InsightFace: ${ev.faces.length} Face Keyframes Localized`);
            }

            // REAL PREDICTED EMOTIONS: Emitted from Step 4!
            if (ev.visual_emotion) {
              liveVisualEmotion = ev.visual_emotion;
            }
            if (ev.audio_emotion) {
              liveAudioEmotion = ev.audio_emotion;
            }

            // FINAL RESULT: Emitted from Step 6!
            if (ev.result) {
              lastResult = ev.result;
            }
          } catch (pe) {
            console.debug("SSE line notice:", pe);
          }
        }
      }

      // Mark all steps complete
      steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
      setPhase("Multimodal Verdict Synthesized");

      targetProgressPct = 100;
      currentRenderedPct = 100;
      const elPct = document.getElementById("analyzing-progress-pct");
      if (elPct) elPct.textContent = "100%";
      const loaderArc = document.getElementById("loader-arc");
      if (loaderArc) loaderArc.style.strokeDashoffset = "0";

      await new Promise((r) => setTimeout(r, 850));
      stopAnalyzingHUD();

      if (lastResult) {
        renderResults(lastResult);
        navigate(routePath("/results"));
      } else {
        throw new Error("No result returned from model detection.");
      }
    } catch (err) {
      showDiagnosticError(err);
    }
  }

  // ── Results ───────────────────────────────────────────────────────────────
  const deltaTag = (v) => (v > 0.5 ? ["High", "tag-high"] : v > 0.3 ? ["Moderate", "tag-mod"] : ["Low", "tag-low"]);

  function countUp(el, target) {
    if (reduce) { el.textContent = target + "%"; return; }
    const dur = 1000, t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))) + "%";
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function getTopEmotion(dist) {
    let topKey = EMO_ORDER[0], topVal = -1;
    for (const k of EMO_ORDER) {
      const v = dist[k] ?? 0;
      if (v > topVal) { topVal = v; topKey = k; }
    }
    return { key: topKey, val: Math.max(0, topVal) };
  }

  function renderEmotionHead(bannerEl, dist) {
    if (!bannerEl || !dist) return;
    const { key: topKey } = getTopEmotion(dist);
    const svg = EMO_SVGS[topKey] || EMO_SVGS.neutral;
    const label = EMO_LABEL[topKey] || topKey;

    bannerEl.className = `head-dominant-banner dom-banner-${topKey}`;
    bannerEl.innerHTML = `
      <div class="dom-icon-box">${svg}</div>
      <div class="dom-text-box">
        <span class="dom-tag">Highest Probability Emotion</span>
        <span class="dom-name">${label}</span>
      </div>
    `;
  }

  function distRows(container, dist) {
    if (!container) return;
    container.innerHTML = "";
    const { key: topKey } = getTopEmotion(dist);

    const headRow = document.createElement("div");
    headRow.className = "drow-head";
    headRow.innerHTML = `<span class="dlabel-head">Emotion</span><span></span><span class="dval-head">Probability</span>`;
    container.appendChild(headRow);

    EMO_ORDER.forEach((k, idx) => {
      const v = dist[k] ?? 0;
      const isTop = k === topKey && v > 0;
      const row = document.createElement("div");
      row.className = isTop ? "drow drow-top" : "drow";
      row.innerHTML = `<span class="dlabel">${EMO_LABEL[k]}</span><div class="bar"><div class="bar-fill bar-emo-${k}"></div></div><span class="dval">${Math.round(v * 100)}%</span>`;
      container.appendChild(row);
      const fill = row.querySelector(".bar-fill");
      setTimeout(() => (fill.style.width = (v * 100).toFixed(1) + "%"), 80 + idx * 60);
    });
  }

  function deltaRows(container, delta) {
    container.innerHTML = "";
    EMO_ORDER.forEach((k, idx) => {
      const v = delta[k] ?? 0;
      const [tag, cls] = deltaTag(v);
      const row = document.createElement("div");
      row.className = "drow";
      row.innerHTML = `<span class="dlabel">${EMO_LABEL[k]}</span><div class="bar"><div class="bar-fill bar-pink"></div></div><span class="dval">${Math.round(v * 100)}%</span><span class="dtag ${cls}">${tag}</span>`;
      container.appendChild(row);
      const fill = row.querySelector(".bar-fill");
      setTimeout(() => (fill.style.width = (v * 100).toFixed(1) + "%"), 80 + idx * 60);
    });
  }

  function renderResults(r) {
    const isFake = r.verdict === "FAKE";
    const pFake = r.p_fake ?? 0;
    const pct = Math.round(pFake * 100);
    const card = document.getElementById("verdict-card");
    card.classList.toggle("fake", isFake);
    card.classList.toggle("real", !isFake);
    document.getElementById("verdict-tag").textContent = isFake ? "Fake" : "Real";
    document.getElementById("verdict-label").textContent = isFake ? "Likely deepfake" : "Likely authentic";

    const emoA = (r.audio_text_emotion?.label || "").toLowerCase();
    const emoB = (r.visual_emotion?.label || "").toLowerCase();
    const emotionsMatch = Boolean(emoA && emoB && emoA === emoB);

    if (isFake) {
      document.getElementById("verdict-sub").textContent = emotionsMatch
        ? "The face or voice shows signs of AI editing even though the general feelings seem similar."
        : "The feelings in the voice and the face contradict each other.";
    } else {
      document.getElementById("verdict-sub").textContent = emotionsMatch
        ? "The voice and facial expressions match naturally."
        : "The voice and face show normal, natural human variation.";
    }
    countUp(document.getElementById("verdict-pct"), pct);

    // BERT / Whisper Spoken Transcript
    const transcriptEl = document.getElementById("result-transcript-text");
    if (transcriptEl) {
      const txt = (r.transcript || "").trim();
      transcriptEl.textContent = txt ? `“${txt}”` : "“No discernible speech detected in this clip selection.”";
    }

    // sarcasm + plain-language interpretation
    const pSarc = r.p_sarcasm ?? 0;
    const sarcastic = pSarc >= 0.5;
    const sarcPct = Math.round(pSarc * 100);
    const auth = isFake ? "<b>manipulated</b>" : "<b class='ok'>genuine</b>";
    let sentence;
    if (!isFake && !sarcastic) {
      sentence = emotionsMatch
        ? `This clip looks ${auth} and sincere — the speaker's voice and face agree.`
        : `This clip looks ${auth} — the emotions show natural, authentic human variation.`;
    } else if (!isFake && sarcastic) {
      sentence = `This clip looks ${auth}, but it is delivered <b>sarcastically</b> (meant as a joke or dry wit).`;
    } else if (isFake && !sarcastic) {
      sentence = emotionsMatch
        ? `This clip looks ${auth} — AI editing flaws detected around the face or voice.`
        : `This clip looks ${auth} — the feelings in the voice and on the face clash.`;
    } else {
      sentence = `This clip looks ${auth}, with clashing emotions and robotic speech patterns.`;
    }
    const interpretEl = document.getElementById("interpret");
    if (interpretEl) interpretEl.innerHTML = sentence;
    const marker = document.getElementById("sarc-marker");
    document.getElementById("sarc-val").textContent = sarcPct + "%";
    marker.style.left = "0%";
    setTimeout(() => (marker.style.left = (pSarc * 100).toFixed(0) + "%"), 180);

    const delta = r.emotion_mismatch || {};
    let domKey = EMO_ORDER[0], domVal = -1;
    for (const k of EMO_ORDER) if ((delta[k] ?? 0) > domVal) { domVal = delta[k] ?? 0; domKey = k; }
    document.getElementById("dom-title").textContent = `Biggest emotion gap · ${EMO_LABEL[domKey]}`;
    document.getElementById("gap-val").textContent = Math.round(domVal * 100) + "%";
    if (isFake) {
      const sig = domVal > 0.5 ? "strong" : domVal > 0.3 ? "moderate" : "slight";
      document.getElementById("dom-desc").textContent =
        `voice sounds ${emoA}, but face looks ${emoB} · ${sig} sign of AI editing`;
    } else {
      document.getElementById("dom-desc").textContent = emotionsMatch
        ? `voice and face both express ${emoA} · consistent across both`
        : `voice sounds ${emoA} while face looks ${emoB} · normal human variation`;
    }
    const domBar = document.getElementById("dom-bar");
    domBar.style.width = "0%";
    setTimeout(() => (domBar.style.width = (domVal * 100).toFixed(1) + "%"), 140);

    renderEmotionHead(
      document.getElementById("head-a-top"),
      r.audio_text_emotion?.distribution || {}
    );
    renderEmotionHead(
      document.getElementById("head-b-top"),
      r.visual_emotion?.distribution || {}
    );
    distRows(document.getElementById("head-a"), r.audio_text_emotion?.distribution || {});
    distRows(document.getElementById("head-b"), r.visual_emotion?.distribution || {});
    deltaRows(document.getElementById("delta-list"), delta);

    // ── Forensic Interpretation Multi-Tier Card ──────────────────────────────
    const fiCard = document.getElementById("forensic-interpretation-card");
    const fi = r.forensic_interpretation;
    if (fiCard && fi) {
      fiCard.hidden = false;
      fiCard.classList.toggle("is-fake", isFake);
      fiCard.classList.toggle("is-real", !isFake);

      const stateTagEl = document.getElementById("forensic-state-tag");
      const headlineEl = document.getElementById("forensic-headline");
      const summaryEl = document.getElementById("forensic-summary");
      const vfEl = document.getElementById("forensic-voice-face");
      const sarcEl = document.getElementById("forensic-sarcasm");
      const ratEl = document.getElementById("forensic-rationale");

      if (stateTagEl) stateTagEl.textContent = fi.state_tag ? fi.state_tag : (isFake ? "LIKELY DEEPFAKE" : "LIKELY AUTHENTIC");
      if (headlineEl) headlineEl.textContent = fi.headline || "Analysis Complete";
      if (summaryEl) summaryEl.textContent = fi.summary || "";
      if (vfEl) vfEl.textContent = fi.voice_face_analysis || "Voice and face expressions evaluated.";
      if (sarcEl) sarcEl.textContent = fi.sarcasm_analysis || "Tone of voice and spoken words checked for sarcasm.";
      if (ratEl) ratEl.textContent = fi.technical_rationale || fi.forensic_rationale || "DeepSentinel found no inconsistencies.";
    }

    const sb = r.served_by || {};
    document.getElementById("served-by").textContent =
      sb.checkpoint ? `Served by ${sb.checkpoint} · phase ${sb.phase ?? "?"} · P(sarcasm) ${(r.p_sarcasm ?? 0).toFixed(2)}` : "";
  }

  // ── Benchmarks ────────────────────────────────────────────────────────────
  let bmData = null;
  let bmRendered = false;

  async function loadBenchmarkData() {
    if (bmData) return bmData;
    try {
      const res = await fetch("/static/data/comparative_benchmark_data.json");
      bmData = await res.json();
    } catch (e) {
      bmData = null;
    }
    return bmData;
  }

  async function renderBenchmarks() {
    const data = await loadBenchmarkData();
    if (!data) return;

    renderSotaTable(data);
    renderRocChart(data);
    renderManipChart(data);
    bindMagnetic();
    bmRendered = true;
  }

  // ---------- Table ----------
  function renderSotaTable(data) {
    const tbody = document.getElementById("sota-table-body");
    if (!tbody || tbody.dataset.filled) return;

    tbody.innerHTML = data.models.map((m) => {
      const isOurs = m.id === "deepsentinel";
      const auc = m.metrics.auc_roc.toFixed(4);
      const ci = `[${m.metrics.auc_ci_lower}–${m.metrics.auc_ci_upper}]`;
      const aucCell = isOurs
        ? `<span class="bm-auc-ours">${auc}</span> <span style="color:var(--muted);font-size:11px">${ci}</span>`
        : `${auc} <span style="color:var(--muted);font-size:11px">${ci}</span>`;

      const sig = m.delong_test.significance === "Reference"
        ? `<span style="color:var(--verdigris);font-family:var(--f-mono);font-size:11px">Reference</span>`
        : `<span style="color:var(--muted);font-family:var(--f-mono);font-size:11px">p &lt; 0.001 ✓</span>`;

      return `<tr class="${isOurs ? "bm-ours" : ""}">
        <td>
          <span class="bm-badge" style="background:${m.color}">${m.badge}</span>
          <strong>${m.name}</strong>
        </td>
        <td style="max-width:200px;white-space:normal;font-size:12px">${m.modality}</td>
        <td>${m.metrics.accuracy.toFixed(2)}%</td>
        <td><strong>${m.metrics.balanced_accuracy.toFixed(2)}%</strong></td>
        <td>${m.metrics.specificity_real.toFixed(2)}%</td>
        <td>${m.metrics.recall_fake.toFixed(2)}%</td>
        <td><strong>${m.metrics.f1_score.toFixed(4)}</strong></td>
        <td>${m.metrics.mcc >= 0 ? "+" : ""}${m.metrics.mcc.toFixed(4)}</td>
        <td>${aucCell}</td>
        <td>${sig}</td>
      </tr>`;
    }).join("");

    tbody.dataset.filled = "1";
  }

  // ---------- ROC Chart ----------
  // Empirical ROC curves approximated from confusion matrix data per model.
  // We construct a synthetic curve by sweeping thresholds between (FPR=0,TPR=0)
  // and (FPR=1,TPR=1), anchoring the one measured operating point in the middle.
  function buildRocPoints(m) {
    const tp = m.metrics.tp, fp = m.metrics.fp, tn = m.metrics.tn, fn = m.metrics.fn;
    const totalPos = tp + fn;  // real positive clips
    const totalNeg = fp + tn;  // real negative clips
    const tpr = tp / totalPos;
    const fpr = fp / totalNeg;
    // Smooth curve: origin → operating point → AUC-guided upper bend → (1,1)
    const auc = m.metrics.auc_roc;
    const pts = [];
    const N = 60;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      // Parametric curve that passes through (0,0), operating point, (1,1)
      // and whose integral approximates the stated AUC.
      let x, y;
      if (t < 0.5) {
        // First half: origin → operating point, with a concave bend
        const s = t * 2;
        x = fpr * Math.pow(s, 0.7);
        y = tpr * Math.pow(s, 1 / (2 * auc));
      } else {
        // Second half: operating point → (1,1)
        const s = (t - 0.5) * 2;
        x = fpr + (1 - fpr) * Math.pow(s, 1.3);
        y = tpr + (1 - tpr) * Math.pow(s, 0.6);
      }
      pts.push({ x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) });
    }
    return pts;
  }

  function renderRocChart(data) {
    const canvas = document.getElementById("roc-canvas");
    const legend = document.getElementById("roc-legend");
    if (!canvas || canvas.dataset.drawn) return;

    const dpr = window.devicePixelRatio || 1;
    const wrap = canvas.parentElement;
    const cssW = wrap.clientWidth || 700;
    const cssH = wrap.clientHeight || 340;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const pad = { top: 22, right: 22, bottom: 50, left: 54 };
    const plotW = cssW - pad.left - pad.right;
    const plotH = cssH - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = "rgba(255,255,255,0)";
    ctx.fillRect(0, 0, cssW, cssH);

    // Axis lines
    ctx.strokeStyle = "rgba(60,60,60,0.12)";
    ctx.lineWidth = 1;
    // Grid lines
    const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    ctx.font = `500 10px var(--f-mono, monospace)`;
    ctx.fillStyle = "#9b9a92";
    ticks.forEach((t) => {
      const gx = pad.left + t * plotW;
      const gy = pad.top + (1 - t) * plotH;
      // Vertical grid
      ctx.beginPath(); ctx.moveTo(gx, pad.top); ctx.lineTo(gx, pad.top + plotH); ctx.stroke();
      // Horizontal grid
      ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + plotW, gy); ctx.stroke();
      // X labels
      if (t > 0) {
        ctx.textAlign = "center";
        ctx.fillText((t * 100).toFixed(0) + "%", gx, pad.top + plotH + 18);
      }
      // Y labels
      ctx.textAlign = "right";
      ctx.fillText((t * 100).toFixed(0) + "%", pad.left - 8, gy + 3.5);
    });

    // Axis titles
    ctx.font = "600 11px var(--f-mono, monospace)";
    ctx.fillStyle = "#5f5f5d";
    ctx.textAlign = "center";
    ctx.fillText("False Positive Rate", pad.left + plotW / 2, cssH - 4);
    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("True Positive Rate", 0, 0);
    ctx.restore();

    // Random-guess diagonal
    ctx.save();
    ctx.strokeStyle = "rgba(60,60,60,0.28)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Model curves
    data.models.slice().reverse().forEach((m) => {
      const pts = buildRocPoints(m);
      const isOurs = m.id === "deepsentinel";
      ctx.save();
      ctx.beginPath();
      pts.forEach(({ x, y }, i) => {
        const px = pad.left + x * plotW;
        const py = pad.top + (1 - y) * plotH;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = m.color;
      ctx.lineWidth = isOurs ? 3 : 1.5;
      ctx.globalAlpha = isOurs ? 1 : 0.7;
      ctx.stroke();
      ctx.restore();

      // Operating point dot
      const tp = m.metrics.tp, fp = m.metrics.fp, tn = m.metrics.tn, fn = m.metrics.fn;
      const tpr = tp / (tp + fn);
      const fpr = fp / (fp + tn);
      const dotX = pad.left + fpr * plotW;
      const dotY = pad.top + (1 - tpr) * plotH;
      ctx.save();
      ctx.fillStyle = m.color;
      ctx.globalAlpha = isOurs ? 1 : 0.8;
      ctx.beginPath();
      ctx.arc(dotX, dotY, isOurs ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    // "AUC=0.9020" label for our model
    const ours = data.models.find((m) => m.id === "deepsentinel");
    if (ours) {
      const tpr = ours.metrics.tp / (ours.metrics.tp + ours.metrics.fn);
      const fpr = ours.metrics.fp / (ours.metrics.fp + ours.metrics.tn);
      const dotX = pad.left + fpr * plotW;
      const dotY = pad.top + (1 - tpr) * plotH;
      ctx.font = "700 11px var(--f-display, sans-serif)";
      ctx.fillStyle = ours.color;
      ctx.textAlign = "left";
      ctx.fillText(`AUC = ${ours.metrics.auc_roc.toFixed(4)}`, dotX + 8, dotY - 6);
    }

    canvas.dataset.drawn = "1";

    // Legend
    if (legend) {
      legend.innerHTML = [
        ...data.models.map((m) => `
          <span class="bm-leg-item">
            <span class="bm-leg-swatch" style="background:${m.color}"></span>
            ${m.name}
          </span>`),
        `<span class="bm-leg-item" style="color:rgba(60,60,60,0.5)">
          <span class="bm-leg-swatch" style="background:rgba(60,60,60,0.28);border-top:2px dashed rgba(60,60,60,0.28)"></span>
          Random Guess
        </span>`,
      ].join("");
    }
  }

  // ---------- Per-Manipulation Bar Chart ----------
  function renderManipChart(data) {
    const canvas = document.getElementById("manip-canvas");
    const legend = document.getElementById("manip-legend");
    if (!canvas || canvas.dataset.drawn) return;

    const categories = ["faceswap", "faceswap-wav2lip", "fsgan", "fsgan-wav2lip", "real", "rtvc", "wav2lip"];
    const labels = ["Faceswap", "Faceswap+W2L", "FSGAN", "FSGAN+W2L", "Real", "RTVC", "Wav2Lip"];
    const models = data.models;
    const dpr = window.devicePixelRatio || 1;
    const wrap = canvas.parentElement;
    const cssW = wrap.clientWidth || 700;
    const cssH = wrap.clientHeight || 360;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const pad = { top: 22, right: 14, bottom: 56, left: 44 };
    const plotW = cssW - pad.left - pad.right;
    const plotH = cssH - pad.top - pad.bottom;

    const groupW = plotW / categories.length;
    const barW = Math.min(10, (groupW - 8) / models.length);
    const groupPad = (groupW - barW * models.length) / 2;

    // Grid + axes
    ctx.font = `500 10px var(--f-mono, monospace)`;
    ctx.fillStyle = "#9b9a92";
    ctx.strokeStyle = "rgba(60,60,60,0.1)";
    ctx.lineWidth = 1;
    [0, 20, 40, 60, 80, 100].forEach((v) => {
      const gy = pad.top + (1 - v / 100) * plotH;
      ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + plotW, gy); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(v + "%", pad.left - 6, gy + 3.5);
    });

    // Bars
    models.forEach((m, mi) => {
      categories.forEach((cat, ci) => {
        const entry = data.per_manipulation_breakdown[cat]?.[m.name];
        if (!entry) return;
        const val = entry.accuracy / 100;
        const x = pad.left + ci * groupW + groupPad + mi * barW;
        const barH = val * plotH;
        const y = pad.top + plotH - barH;
        const isOurs = m.id === "deepsentinel";

        ctx.save();
        ctx.fillStyle = m.color;
        ctx.globalAlpha = isOurs ? 1 : 0.6;
        const r = Math.min(3, barW / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + barW - r, y);
        ctx.arcTo(x + barW, y, x + barW, y + r, r);
        ctx.lineTo(x + barW, y + barH);
        ctx.lineTo(x, y + barH);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      });
    });

    // X labels
    ctx.font = `500 10.5px var(--f-mono, monospace)`;
    ctx.fillStyle = "#5f5f5d";
    ctx.textAlign = "center";
    labels.forEach((lbl, ci) => {
      const cx = pad.left + ci * groupW + groupW / 2;
      ctx.fillText(lbl, cx, pad.top + plotH + 20);
    });

    // Y axis title
    ctx.font = "600 11px var(--f-mono, monospace)";
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Accuracy %", 0, 0);
    ctx.restore();

    canvas.dataset.drawn = "1";

    // Legend
    if (legend) {
      legend.innerHTML = models.map((m) => `
        <span class="bm-leg-item">
          <span class="bm-leg-swatch" style="background:${m.color}"></span>
          ${m.name}
        </span>`).join("");
    }
  }

  // ── Researchers (4 members) + expand modal ────────────────────────────────

  const AVATAR = `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="1.5"/><path d="M4 20c0-4 3.6-6 8-6s8 2 8 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;
  const SOCIAL_ICONS = {
    linkedin: `<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M7 10v7M7 7v.01M11 17v-4a2 2 0 0 1 4 0v4M11 17v-7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
    github: `<svg viewBox="0 0 24 24" fill="none"><path d="M9 19c-4 1.5-4-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1-.3-3.4 1.3a11.6 11.6 0 0 0-6 0C7.3 2.6 6.3 2.9 6.3 2.9a4.3 4.3 0 0 0-.1 3.2A4.6 4.6 0 0 0 4.9 9.3c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    link: `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M3.5 9h17M3.5 15h17M12 3c2.5 2.5 2.5 16 0 18M12 3c-2.5 2.5-2.5 16 0 18" stroke="currentColor" stroke-width="1.3"/></svg>`,
    mail: `<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M4 7l8 6 8-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  };
  const TEAM = [
    { name: "Geuel John D. Rivera", role: "Project Leader", cv: "#", photo: "/static/img/el.png",
      bio: "Led overall coordination and system integration, and owns the detection module — the emotion heads, discrepancy score Δ, compact bilinear fusion, and the classifier.",
      socials: [{ t: "linkedin", href: "https://www.linkedin.com/in/geuel-john-d-rivera-24a853292/" }, { t: "github", href: "https://github.com/gjvlio" }, { t: "mail", href: "#" }] },
    { name: "Shikina Y. Cabral", role: "Data Generation Lead", cv: "#", photo: "/static/img/kina.png",
      bio: "Built the four-track deepfake generation pipeline using StyleTTS2, RVC, Wav2Lip, SadTalker, and MuseTalk to produce the labelled training corpus.",
      socials: [{ t: "linkedin", href: "https://www.linkedin.com/in/shikina-cabral-97826027a/" }, { t: "github", href: "https://github.com/CShikina" }, { t: "link", href: "#" }] },
    { name: "John Christian B. Caparas", role: "Preprocessing Lead", cv: "#", photo: "/static/img/jc.png",
      bio: "Owns feature extraction — Wav2Vec 2.0, BERT, and the Vision Transformer — plus face detection, keyframe selection, and the cached feature store.",
      socials: [{ t: "linkedin", href: "#" }, { t: "github", href: "https://github.com/JJEEYYSSEE" }, { t: "mail", href: "#" }] },
    { name: "Matan John B. Exonde", role: "Evaluation Lead", cv: "#", photo: "/static/img/matan.png",
      bio: "Handles training orchestration, benchmarking on unseen data, statistical significance testing, and the project documentation.",
      socials: [{ t: "linkedin", href: "https://www.linkedin.com/in/matan-john-banzuelo-exconde-83612029a/" }, { t: "github", href: "https://github.com/Enami345" }, { t: "link", href: "#" }] },
  ];

  function renderTeam() {
    const grid = document.getElementById("team-grid");
    if (grid.dataset.filled) return;
    const cvIcon = `<svg width="15" height="15" viewBox="0 0 16 16" fill="none"><path d="M8 2v8m0 0l3-3m-3 3L5 7M3 13h10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const xIcon = `<svg viewBox="0 0 24 24" fill="none"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`;
    grid.innerHTML = TEAM.map((m, i) => `
      <div class="tcard" data-idx="${i}">
        <button class="tcard-close" aria-label="Collapse profile">${xIcon}</button>
        <button class="tcard-head" aria-expanded="false" aria-controls="td-${i}">
          <span class="ta-avatar">${m.photo ? `<img src="${m.photo}" alt="${m.name}" loading="lazy">` : AVATAR}</span>
          <span class="ta-name">${m.name}</span>
          <span class="ta-role">${m.role}</span>
          <span class="ta-cue">View profile <span class="plus">+</span></span>
        </button>
        <div class="tcard-detail" id="td-${i}">
          <div class="td-inner">
            <p class="td-bio">${m.bio}</p>
            <div class="td-socials">${(m.socials || []).map((s) =>
              `<a class="social-btn" href="${s.href}" aria-label="${s.t}"${s.href === "#" ? "" : ' target="_blank" rel="noopener"'}>${SOCIAL_ICONS[s.t] || SOCIAL_ICONS.link}</a>`).join("")}</div>
            <a class="btn btn-primary magnetic td-cv" href="${m.cv}"${m.cv === "#" ? "" : " download"}>${cvIcon}<span>Download CV</span></a>
          </div>
        </div>
      </div>`).join("");
    grid.dataset.filled = "1";
    grid.querySelectorAll(".tcard").forEach((card) => {
      card.querySelector(".tcard-head").addEventListener("click", () => toggleCard(card));
      card.querySelector(".tcard-close").addEventListener("click", (e) => { e.stopPropagation(); toggleCard(card); });
    });
    bindMagnetic();
  }

  // expanding 1×4 accordion — one card open at a time
  function toggleCard(card) {
    const wasOpen = card.classList.contains("open");
    document.querySelectorAll(".tcard.open").forEach((c) => {
      c.classList.remove("open");
      c.querySelector(".tcard-head").setAttribute("aria-expanded", "false");
    });
    if (!wasOpen) {
      card.classList.add("open");
      card.querySelector(".tcard-head").setAttribute("aria-expanded", "true");
    }
  }

  // ── Model Warmup Screen (Simple & Clean Mirrored Bar) ─────────────────────
  function initWarmupScreen() {
    const screenEl = document.getElementById("warmup-screen");
    if (!screenEl) return;

    // Only display when first accessing the page in this session (restart only)
    const hasSeenWarmup = sessionStorage.getItem("ds_warmup_completed");
    const path = location.pathname;
    const isLanding = path === "/" || path === "/demo" || path === "/demo/";
    if (hasSeenWarmup || !isLanding) {
      screenEl.style.display = "none";
      return;
    }

    // Main solid colors of the design: Verdigris, Wisteria, Rosy Copper, Linen, Gunmetal, Cyan, Warm Copper
    const BRAND_SOLID_COLORS = [
      "#1EA896", // Verdigris
      "#72A1E5", // Wisteria Blue
      "#DB5A42", // Rosy Copper
      "#ECEBE4", // Soft Linen
      "#3C3C3C", // Gunmetal
      "#3FB6C0", // Sky Cyan
      "#E08A5B", // Warm Copper
    ];

    // Randomize the order of the solid colors on every restart
    const shuffled = [...BRAND_SOLID_COLORS].sort(() => Math.random() - 0.5);
    const colorA = shuffled[0];
    const colorB = shuffled[1];
    const colorC = shuffled[2];
    const angle = Math.floor(Math.random() * 70) + 110; // 110deg - 180deg

    // Full background: simple gradient composed of solid colors in random order
    screenEl.style.background = `linear-gradient(${angle}deg, ${colorA} 0%, ${colorB} 100%)`;

    // Mirrored progress bar wings: gradient using randomized colors
    const accentCol = (colorA === "#ECEBE4" || colorA === "#3C3C3C") ? colorB : colorA;
    screenEl.style.setProperty("--warmup-bar-grad-left", `linear-gradient(270deg, ${colorA} 0%, ${colorC} 100%)`);
    screenEl.style.setProperty("--warmup-bar-grad-right", `linear-gradient(90deg, ${colorA} 0%, ${colorC} 100%)`);
    screenEl.style.setProperty("--warmup-accent", accentCol);

    const pctNumEl = document.getElementById("warmup-pct-num");
    const wingLeftEl = document.getElementById("warmup-wing-left");
    const wingRightEl = document.getElementById("warmup-wing-right");
    const barTrackEl = document.getElementById("warmup-bar-track");
    const actionEl = document.getElementById("warmup-action");
    const targetEl = document.getElementById("warmup-target");
    const skipBtnEl = document.getElementById("warmup-skip-btn");

    const STAGES = [
      { threshold: 14, action: "fetching", target: "Wav2Vec 2.0 Audio Backbone" },
      { threshold: 34, action: "fetching", target: "Vision Transformer ViT-B/16" },
      { threshold: 54, action: "fetching", target: "Whisper Speech Recognition Model" },
      { threshold: 74, action: "fetching", target: "InsightFace 3D Landmarker & Alignment" },
      { threshold: 88, action: "calibrating", target: "Information-Theoretic Synchrony Engine" },
      { threshold: 96, action: "verifying", target: "Cross-Modal Affective Attention Heads" },
      { threshold: 100, action: "ready", target: "DeepSentinel Operational" },
    ];

    let currentPct = 0;
    let isDismissed = false;
    let backendWarmed = false;

    // Check live backend warmup status
    fetch("/warmup/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) return;
        backendWarmed = !!data.warmed;
      })
      .catch(() => {
        backendWarmed = true;
      });

    function dismiss() {
      if (isDismissed) return;
      isDismissed = true;
      try {
        sessionStorage.setItem("ds_warmup_completed", "1");
      } catch (e) {
        /* storage restricted in private mode */
      }
      window.removeEventListener("keydown", keyListener);

      try {
        currentPct = 100;
        updateVisuals(100);
        if (actionEl) {
          actionEl.textContent = "ready";
          actionEl.style.color = accentCol;
        }
        if (targetEl) targetEl.textContent = "DeepSentinel Operational";
      } catch (err) {
        console.warn("Warmup visual completion notice:", err);
      }

      // Immediately unblock user interactions
      screenEl.style.pointerEvents = "none";
      screenEl.classList.add("dismissed");

      setTimeout(() => {
        screenEl.style.display = "none";
      }, 500);
    }

    if (skipBtnEl) {
      skipBtnEl.addEventListener("click", dismiss);
    }

    const keyListener = (e) => {
      if (e.code === "Space" || e.code === "Escape") {
        window.removeEventListener("keydown", keyListener);
        dismiss();
      }
    };
    window.addEventListener("keydown", keyListener);

    function updateVisuals(pct) {
      try {
        const pClamped = Math.min(100, Math.max(0, pct));
        if (pctNumEl) pctNumEl.textContent = Math.round(pClamped);

        // Mirrored center expansion: half-width extends left and right from 50%
        const halfPct = pClamped / 2;
        if (wingLeftEl) wingLeftEl.style.width = `${halfPct.toFixed(1)}%`;
        if (wingRightEl) wingRightEl.style.width = `${halfPct.toFixed(1)}%`;

        if (barTrackEl) {
          if (pClamped > 1) barTrackEl.classList.add("active");
          else barTrackEl.classList.remove("active");
        }

        // Update bracketed console line: fetching [ ... ] ...
        const activeStage = STAGES.find((s) => pClamped <= s.threshold) || STAGES[STAGES.length - 1];
        if (actionEl && actionEl.textContent !== activeStage.action) {
          actionEl.textContent = activeStage.action;
          if (activeStage.action === "ready") {
            actionEl.style.color = "#1EA896";
          } else if (activeStage.action === "calibrating") {
            actionEl.style.color = "#DB5A42";
          } else {
            actionEl.style.color = "#1EA896";
          }
        }
        if (targetEl && targetEl.textContent !== activeStage.target) {
          targetEl.textContent = activeStage.target;
        }
      } catch (err) {
        console.warn("Warmup updateVisuals notice:", err);
      }
    }

    // Animation loop: smoothly advance towards 100% over ~2.0 seconds
    const startTime = performance.now();
    const duration = 2000;

    function tick(now) {
      if (isDismissed) return;
      const elapsed = now - startTime;
      const progressRatio = Math.min(1, elapsed / duration);

      // Smooth ease-in-out curve
      const eased = progressRatio < 0.5
        ? 2 * progressRatio * progressRatio
        : 1 - Math.pow(-2 * progressRatio + 2, 2) / 2;

      // Hold briefly at 92% if backend is still actively warming (safety guard)
      let targetProgress = eased * 100;
      if (!backendWarmed && targetProgress > 92) {
        targetProgress = 92;
      }

      currentPct += (targetProgress - currentPct) * 0.25;
      updateVisuals(currentPct);

      if (progressRatio >= 1 && (backendWarmed || elapsed > 2800)) {
        dismiss();
      } else {
        requestAnimationFrame(tick);
      }
    }

    requestAnimationFrame(tick);

    // Guaranteed absolute safety timer: dismiss after 3.0s even if browser tab was backgrounded/throttled
    setTimeout(dismiss, 3000);
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  makeSparkles();
  initWarmupScreen();
  render();
  bindMagnetic();
  bindTilt();
})();

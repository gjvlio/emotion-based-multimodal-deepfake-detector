/* DeepSentinel SPA — routing, detect flow, and the motion layer. Vanilla JS. */
(() => {
  "use strict";

  const EMO_ORDER = ["angry", "happy", "sad", "neutral", "fear", "disgust"];
  const EMO_LABEL = { angry: "Angry", happy: "Happy", sad: "Sad", neutral: "Neutral", fear: "Fearful", disgust: "Disgust" };

  const ROUTES = {
    "/": "landing", "/upload": "upload", "/analyzing": "analyzing", "/results": "results",
    "/about": "about-thesis", "/about/thesis": "about-thesis", "/about/researchers": "about-researchers",
  };

  const views = {};
  document.querySelectorAll("[data-view]").forEach((v) => (views[v.dataset.view] = v));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let selectedFile = null;
  let lastResult = null;

  // ── Routing ───────────────────────────────────────────────────────────────
  function navigate(path, replace = false) {
    if (replace) history.replaceState({}, "", path);
    else history.pushState({}, "", path);
    render();
  }

  function render() {
    let path = location.pathname;
    let view = ROUTES[path] || "landing";
    if (view === "analyzing" && !selectedFile) { view = "upload"; history.replaceState({}, "", "/upload"); }
    if (view === "results" && !lastResult) { view = "upload"; history.replaceState({}, "", "/upload"); }

    Object.values(views).forEach((v) => v.classList.remove("active"));
    const el = views[view] || views.landing;
    el.classList.add("active");
    el.classList.remove("view-enter"); void el.offsetWidth; el.classList.add("view-enter");
    window.scrollTo({ top: 0 });

    document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
    if (path === "/") document.querySelector('.nav-link[href="/"]')?.classList.add("active");
    if (path.startsWith("/about")) document.querySelector(".nav-dropdown-toggle")?.classList.add("active");

    document.body.classList.toggle("no-scroll", view === "landing" || view === "about-thesis");
    if (view === "upload") resetUpload();
    if (view === "about-researchers") renderTeam();
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
    navigate(a.getAttribute("href"));
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

  // ── Upload ────────────────────────────────────────────────────────────────
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dzEmpty = dropzone.querySelector(".dz-empty");
  const dzFile = dropzone.querySelector(".dz-file");
  const dzFileName = dropzone.querySelector(".dz-file-name");
  const runBtn = document.getElementById("run-btn");
  const uploadError = document.getElementById("upload-error");
  const ALLOWED = [".mp4", ".mov", ".webm"];
  const fmtSize = (b) => (b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB");

  function pickFile(file) {
    if (!file) return;
    if (!ALLOWED.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      return showError(`Unsupported file. Use ${ALLOWED.join(", ")}.`);
    }
    hideError();
    selectedFile = file;
    dzFileName.textContent = `${file.name} · ${fmtSize(file.size)}`;
    dzEmpty.hidden = true; dzFile.hidden = false; runBtn.hidden = false;
  }
  const showError = (m) => { uploadError.textContent = m; uploadError.hidden = false; };
  const hideError = () => (uploadError.hidden = true);

  function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    dzEmpty.hidden = false;
    dzFile.hidden = true;
    runBtn.hidden = true;
    hideError();
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
  runBtn.addEventListener("click", runAnalysis);

  // ── Analyze flow ──────────────────────────────────────────────────────────
  async function runAnalysis() {
    if (!selectedFile) return;
    document.getElementById("analyzing-file").textContent = `${selectedFile.name} · ${fmtSize(selectedFile.size)}`;
    navigate("/analyzing");
    const stepsDone = animateSteps();
    const form = new FormData();
    form.append("file", selectedFile);
    try {
      const res = await fetch("/detect", { method: "POST", body: form });
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || `Server error (${res.status})`); }
      lastResult = await res.json();
      await stepsDone();
      renderResults(lastResult);
      navigate("/results");
    } catch (err) {
      navigate("/upload");
      showError(err.message || "Analysis failed.");
    }
  }

  function animateSteps() {
    const steps = [...document.querySelectorAll("#steps li")];
    steps.forEach((s) => s.classList.remove("done", "active"));
    // randomised per-step delays so it reads like real, uneven processing
    const delays = steps.map(() => (reduce ? 110 : 340 + Math.random() * 640));
    let i = 0, cancelled = false, resolveDone;
    const done = new Promise((r) => (resolveDone = r));
    (function next() {
      if (cancelled) return;
      if (i > 0) { steps[i - 1].classList.remove("active"); steps[i - 1].classList.add("done"); }
      if (i < steps.length) { steps[i].classList.add("active"); setTimeout(next, delays[i++]); }
      else resolveDone();
    })();
    return async () => {
      await done;
      cancelled = true;
      steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
    };
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

  function distRows(container, dist, barClass) {
    container.innerHTML = "";
    EMO_ORDER.forEach((k, idx) => {
      const v = dist[k] ?? 0;
      const row = document.createElement("div");
      row.className = "drow";
      row.innerHTML = `<span class="dlabel">${EMO_LABEL[k]}</span><div class="bar"><div class="bar-fill ${barClass}"></div></div><span class="dval">${v.toFixed(2)}</span>`;
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
      row.innerHTML = `<span class="dlabel">${EMO_LABEL[k]}</span><span class="dval" style="text-align:left">${v.toFixed(2)}</span><div class="bar"><div class="bar-fill bar-pink"></div></div><span class="dtag ${cls}">${tag}</span>`;
      container.appendChild(row);
      const fill = row.querySelector(".bar-fill");
      setTimeout(() => (fill.style.width = (v * 100).toFixed(1) + "%"), 80 + idx * 60);
    });
  }

  function renderResults(r) {
    const isFake = r.verdict === "FAKE";
    const pct = Math.round(r.p_fake * 100);
    const card = document.getElementById("verdict-card");
    card.classList.toggle("fake", isFake);
    card.classList.toggle("real", !isFake);
    document.getElementById("verdict-tag").textContent = isFake ? "Fake" : "Real";
    document.getElementById("verdict-label").textContent = isFake ? "Likely deepfake" : "Likely authentic";
    document.getElementById("verdict-sub").textContent = isFake
      ? "Strong emotional mismatch detected across modalities."
      : "Emotions are consistent across audio and visual modalities.";
    countUp(document.getElementById("verdict-pct"), pct);

    // sarcasm + plain-language interpretation
    const pSarc = r.p_sarcasm ?? 0;
    const sarcastic = pSarc >= 0.5;
    const auth = isFake ? "<b>manipulated</b>" : "<b class='ok'>genuine</b>";
    let sentence;
    if (!isFake && !sarcastic) sentence = `This looks ${auth} and sincerely delivered — the voice and the face agree.`;
    else if (!isFake && sarcastic) sentence = `This looks ${auth}, but it is delivered <b>sarcastically</b> — the words may not be meant literally.`;
    else if (isFake && !sarcastic) sentence = `This looks ${auth} — the emotion in the voice and the face do not line up.`;
    else sentence = `This looks ${auth}, and the speech also reads as <b>sarcastic</b>.`;
    document.getElementById("interpret").innerHTML = sentence;
    document.getElementById("sarc-pct").textContent = `${Math.round(pSarc * 100)}% sarcastic`;
    const marker = document.getElementById("sarc-marker");
    marker.style.left = "0%";
    setTimeout(() => (marker.style.left = (pSarc * 100).toFixed(0) + "%"), 180);

    const delta = r.emotion_mismatch || {};
    let domKey = EMO_ORDER[0], domVal = -1;
    for (const k of EMO_ORDER) if ((delta[k] ?? 0) > domVal) { domVal = delta[k] ?? 0; domKey = k; }
    document.getElementById("dom-title").textContent = `Dominant mismatch · ${EMO_LABEL[domKey]}`;
    document.getElementById("dom-delta").textContent = `Δ = ${domVal.toFixed(2)}`;
    const sig = domVal > 0.5 ? "high" : domVal > 0.3 ? "moderate" : "low";
    document.getElementById("dom-desc").textContent =
      `audio reads ${(r.audio_text_emotion?.label || "").toLowerCase()}, face reads ${(r.visual_emotion?.label || "").toLowerCase()} · ${sig} fake signal`;
    const domBar = document.getElementById("dom-bar");
    domBar.style.width = "0%";
    setTimeout(() => (domBar.style.width = (domVal * 100).toFixed(1) + "%"), 140);

    distRows(document.getElementById("head-a"), r.audio_text_emotion?.distribution || {}, "bar-blue");
    distRows(document.getElementById("head-b"), r.visual_emotion?.distribution || {}, "bar-mint");
    deltaRows(document.getElementById("delta-list"), delta);

    const sb = r.served_by || {};
    document.getElementById("served-by").textContent =
      sb.checkpoint ? `Served by ${sb.checkpoint} · phase ${sb.phase ?? "?"} · P(sarcasm) ${(r.p_sarcasm ?? 0).toFixed(2)}` : "";
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

  // ── Boot ──────────────────────────────────────────────────────────────────
  makeSparkles();
  render();
  bindMagnetic();
  bindTilt();
})();

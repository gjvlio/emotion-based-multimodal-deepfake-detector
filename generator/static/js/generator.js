/**
 * Deepfake Generator Studio - Client Controller
 * Provides webcam recording, file drag-and-drop, model execution,
 * synchronized side-by-side comparison, and DeepSentinel verification.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    face: { type: null, file: null, blob: null, presetPath: null, previewUrl: null },
    donor: { type: null, file: null, blob: null, presetPath: null, previewUrl: null },
    mode: 'wav2lip',
    generatedResult: null,
  };

  // Webcam Streams & Recorders
  let faceStream = null;
  let faceRecorder = null;
  let faceChunks = [];
  let faceTimerInterval = null;
  let faceRecordStartTime = 0;

  let donorStream = null;
  let donorRecorder = null;
  let donorChunks = [];
  let donorTimerInterval = null;
  let donorRecordStartTime = 0;

  // DOM Elements
  const btnGenerate = document.getElementById('btn-generate');
  const sectionProgress = document.getElementById('section-progress');
  const sectionResults = document.getElementById('section-results');
  const progressStatusText = document.getElementById('progress-status-text');

  // Showcase elements
  const resVideoFace = document.getElementById('res-video-face');
  const resVideoDonor = document.getElementById('res-video-donor');
  const resVideoFake = document.getElementById('res-video-fake');

  // Sync controls
  const btnSyncPlay = document.getElementById('btn-sync-play');
  const btnSyncPause = document.getElementById('btn-sync-pause');
  const btnSyncRestart = document.getElementById('btn-sync-restart');
  const masterTimeDisplay = document.getElementById('master-time-display');

  // Telemetry elements
  const telemetryModeTag = document.getElementById('telemetry-mode-tag');
  const tTime = document.getElementById('t-time');
  const tFilename = document.getElementById('t-filename');
  const tPath = document.getElementById('t-path');
  const tVideoStream = document.getElementById('t-video-stream');
  const tAudioStream = document.getElementById('t-audio-stream');
  const tSize = document.getElementById('t-size');
  const btnDownloadResult = document.getElementById('btn-download-result');

  // Detector integration elements
  const btnTestDetector = document.getElementById('btn-test-deepfake-sentinel');
  const detectorCard = document.getElementById('detector-live-card');
  const detectorVerdictBadge = document.getElementById('detector-verdict-badge');
  const detectorLiveMsg = document.getElementById('detector-live-msg');
  const btnResetStudio = document.getElementById('btn-reset-studio');

  // --------------------------------------------------------------------------
  // Tab Switching
  // --------------------------------------------------------------------------
  function setupTabs(tabContainerSelector, parentCardSelector) {
    const card = document.querySelector(parentCardSelector);
    if (!card) return;
    const tabs = card.querySelectorAll(tabContainerSelector + ' .tab-pill');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const targetId = tab.getAttribute('data-target');
        card.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) targetPanel.classList.remove('hidden');
      });
    });
  }

  setupTabs('.tab-pill-group', '#card-original-face');
  setupTabs('.tab-pill-group', '#card-donor-material');

  // --------------------------------------------------------------------------
  // Mode Selection
  // --------------------------------------------------------------------------
  const modeCards = document.querySelectorAll('.mode-card');
  modeCards.forEach(card => {
    card.addEventListener('click', () => {
      modeCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const radio = card.querySelector('.mode-radio');
      if (radio) {
        radio.checked = true;
        state.mode = radio.value;
      }
    });
  });

  // --------------------------------------------------------------------------
  // Helper: Supported MIME type for MediaRecorder
  // --------------------------------------------------------------------------
  function getSupportedMimeType() {
    const types = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
      'video/mp4',
    ];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
  }

  // --------------------------------------------------------------------------
  // Helper: Format Time mm:ss
  // --------------------------------------------------------------------------
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  // --------------------------------------------------------------------------
  // Face Webcam Recording
  // --------------------------------------------------------------------------
  const btnStartFaceCam = document.getElementById('btn-start-face-cam');
  const faceCamLive = document.getElementById('face-cam-live');
  const faceCamPlayback = document.getElementById('face-cam-playback');
  const faceCamPlaceholder = document.getElementById('face-cam-placeholder');
  const faceCamControls = document.getElementById('face-cam-controls');
  const btnRecordFace = document.getElementById('btn-record-face');
  const btnStopFace = document.getElementById('btn-stop-face');
  const btnRetakeFace = document.getElementById('btn-retake-face');
  const faceRecordIndicator = document.getElementById('face-record-indicator');
  const faceRecordTimer = document.getElementById('face-record-timer');

  btnStartFaceCam.addEventListener('click', async () => {
    try {
      faceStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 25 } },
        audio: true,
      });
      faceCamLive.srcObject = faceStream;
      faceCamLive.classList.remove('hidden');
      faceCamPlayback.classList.add('hidden');
      faceCamPlaceholder.classList.add('hidden');
      faceCamControls.classList.remove('hidden');
      btnRecordFace.classList.remove('hidden');
      btnStopFace.classList.add('hidden');
      btnRetakeFace.classList.add('hidden');
    } catch (err) {
      alert(`Camera access denied or unavailable: ${err.message}`);
    }
  });

  btnRecordFace.addEventListener('click', () => {
    if (!faceStream) return;
    faceChunks = [];
    const mimeType = getSupportedMimeType();
    faceRecorder = new MediaRecorder(faceStream, mimeType ? { mimeType } : {});

    faceRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) faceChunks.push(e.data);
    };

    faceRecorder.onstop = () => {
      const blob = new Blob(faceChunks, { type: mimeType || 'video/webm' });
      const url = URL.createObjectURL(blob);
      faceCamLive.classList.add('hidden');
      faceCamPlayback.src = url;
      faceCamPlayback.classList.remove('hidden');

      state.face = {
        type: 'blob',
        file: null,
        blob: blob,
        presetPath: null,
        previewUrl: url,
      };

      updateFaceReadyState(true, `Webcam clip recorded (${(blob.size / 1024).toFixed(0)} KB)`);
      btnStopFace.classList.add('hidden');
      btnRetakeFace.classList.remove('hidden');
    };

    faceRecorder.start(250);
    faceRecordStartTime = Date.now();
    faceRecordIndicator.classList.remove('hidden');
    faceTimerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - faceRecordStartTime) / 1000);
      faceRecordTimer.textContent = formatTime(elapsed);
    }, 500);

    btnRecordFace.classList.add('hidden');
    btnStopFace.classList.remove('hidden');
  });

  btnStopFace.addEventListener('click', () => {
    if (faceRecorder && faceRecorder.state !== 'inactive') {
      faceRecorder.stop();
      clearInterval(faceTimerInterval);
      faceRecordIndicator.classList.add('hidden');
    }
  });

  btnRetakeFace.addEventListener('click', () => {
    faceCamPlayback.pause();
    faceCamPlayback.classList.add('hidden');
    faceCamLive.classList.remove('hidden');
    btnRetakeFace.classList.add('hidden');
    btnRecordFace.classList.remove('hidden');
    updateFaceReadyState(false, 'Ready to record again');
  });

  // --------------------------------------------------------------------------
  // Face File Upload (Dropzone)
  // --------------------------------------------------------------------------
  const dropzoneFace = document.getElementById('dropzone-face');
  const fileFace = document.getElementById('file-face');
  const previewFaceUpload = document.getElementById('preview-face-upload');
  const videoFacePreview = document.getElementById('video-face-preview');
  const labelFaceFile = document.getElementById('label-face-file');
  const btnClearFace = document.getElementById('btn-clear-face');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneFace.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneFace.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneFace.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneFace.classList.remove('dragover');
    });
  });

  dropzoneFace.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFaceFile(files[0]);
  });

  fileFace.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFaceFile(e.target.files[0]);
  });

  function handleFaceFile(file) {
    const url = URL.createObjectURL(file);
    videoFacePreview.src = url;
    labelFaceFile.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    dropzoneFace.classList.add('hidden');
    previewFaceUpload.classList.remove('hidden');

    state.face = {
      type: 'file',
      file: file,
      blob: null,
      presetPath: null,
      previewUrl: url,
    };
    updateFaceReadyState(true, `Uploaded: ${file.name}`);
  }

  btnClearFace.addEventListener('click', () => {
    fileFace.value = '';
    videoFacePreview.pause();
    videoFacePreview.src = '';
    previewFaceUpload.classList.add('hidden');
    dropzoneFace.classList.remove('hidden');
    state.face = { type: null, file: null, blob: null, presetPath: null, previewUrl: null };
    updateFaceReadyState(false, 'No face material selected');
  });

  // --------------------------------------------------------------------------
  // Face Presets Selection
  // --------------------------------------------------------------------------
  const presetFaceItems = document.querySelectorAll('#preset-list-face .preset-item');
  const previewFacePreset = document.getElementById('preview-face-preset');
  const videoFacePresetPlayback = document.getElementById('video-face-preset-playback');
  const labelFacePreset = document.getElementById('label-face-preset');
  const btnClearFacePreset = document.getElementById('btn-clear-face-preset');

  presetFaceItems.forEach(item => {
    const btn = item.querySelector('.preset-select-btn');
    btn.addEventListener('click', () => {
      presetFaceItems.forEach(i => i.classList.remove('selected'));
      item.classList.add('selected');
      const path = item.getAttribute('data-path');
      const name = item.querySelector('.preset-name').textContent;

      // Use preset preview
      videoFacePresetPlayback.src = `/outputs/${encodeURIComponent(item.querySelector('.preset-sub').textContent.split(' • ')[0])}`;
      labelFacePreset.textContent = `Preset: ${name}`;
      previewFacePreset.classList.remove('hidden');

      state.face = {
        type: 'preset',
        file: null,
        blob: null,
        presetPath: path,
        previewUrl: videoFacePresetPlayback.src,
      };
      updateFaceReadyState(true, `Selected Preset: ${name}`);
    });
  });

  btnClearFacePreset.addEventListener('click', () => {
    presetFaceItems.forEach(i => i.classList.remove('selected'));
    videoFacePresetPlayback.pause();
    videoFacePresetPlayback.src = '';
    previewFacePreset.classList.add('hidden');
    state.face = { type: null, file: null, blob: null, presetPath: null, previewUrl: null };
    updateFaceReadyState(false, 'No face material selected');
  });

  // --------------------------------------------------------------------------
  // Donor Presets Selection (Default!)
  // --------------------------------------------------------------------------
  const presetDonorItems = document.querySelectorAll('#preset-list-donor .preset-item');
  const previewDonorPreset = document.getElementById('preview-donor-preset');
  const videoDonorPresetPlayback = document.getElementById('video-donor-preset-playback');
  const labelDonorPreset = document.getElementById('label-donor-preset');
  const btnClearDonorPreset = document.getElementById('btn-clear-donor-preset');

  presetDonorItems.forEach(item => {
    const btn = item.querySelector('.preset-select-btn');
    btn.addEventListener('click', () => {
      presetDonorItems.forEach(i => i.classList.remove('selected'));
      item.classList.add('selected');
      const path = item.getAttribute('data-path');
      const name = item.querySelector('.preset-name').textContent;
      const filename = item.querySelector('.preset-sub').textContent.split(' • ')[0];

      // Video / Audio preview
      videoDonorPresetPlayback.src = `/outputs/${encodeURIComponent(filename)}`;
      labelDonorPreset.textContent = `Donor Preset: ${name}`;
      previewDonorPreset.classList.remove('hidden');

      state.donor = {
        type: 'preset',
        file: null,
        blob: null,
        presetPath: path,
        previewUrl: videoDonorPresetPlayback.src,
      };
      updateDonorReadyState(true, `Selected Preset: ${name}`);
    });
  });

  btnClearDonorPreset.addEventListener('click', () => {
    presetDonorItems.forEach(i => i.classList.remove('selected'));
    videoDonorPresetPlayback.pause();
    videoDonorPresetPlayback.src = '';
    previewDonorPreset.classList.add('hidden');
    state.donor = { type: null, file: null, blob: null, presetPath: null, previewUrl: null };
    updateDonorReadyState(false, 'No donor material selected');
  });

  // Auto-select first donor preset if available for instant demo readiness!
  if (presetDonorItems.length > 0) {
    presetDonorItems[0].querySelector('.preset-select-btn').click();
  }

  // --------------------------------------------------------------------------
  // Donor Webcam / Mic Recording
  // --------------------------------------------------------------------------
  const btnStartDonorCam = document.getElementById('btn-start-donor-cam');
  const donorCamLive = document.getElementById('donor-cam-live');
  const donorCamPlayback = document.getElementById('donor-cam-playback');
  const donorCamPlaceholder = document.getElementById('donor-cam-placeholder');
  const donorCamControls = document.getElementById('donor-cam-controls');
  const btnRecordDonor = document.getElementById('btn-record-donor');
  const btnStopDonor = document.getElementById('btn-stop-donor');
  const btnRetakeDonor = document.getElementById('btn-retake-donor');
  const donorRecordIndicator = document.getElementById('donor-record-indicator');
  const donorRecordTimer = document.getElementById('donor-record-timer');

  btnStartDonorCam.addEventListener('click', async () => {
    try {
      donorStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 25 } },
        audio: true,
      });
      donorCamLive.srcObject = donorStream;
      donorCamLive.classList.remove('hidden');
      donorCamPlayback.classList.add('hidden');
      donorCamPlaceholder.classList.add('hidden');
      donorCamControls.classList.remove('hidden');
      btnRecordDonor.classList.remove('hidden');
      btnStopDonor.classList.add('hidden');
      btnRetakeDonor.classList.add('hidden');
    } catch (err) {
      alert(`Mic/Camera access denied or unavailable: ${err.message}`);
    }
  });

  btnRecordDonor.addEventListener('click', () => {
    if (!donorStream) return;
    donorChunks = [];
    const mimeType = getSupportedMimeType();
    donorRecorder = new MediaRecorder(donorStream, mimeType ? { mimeType } : {});

    donorRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) donorChunks.push(e.data);
    };

    donorRecorder.onstop = () => {
      const blob = new Blob(donorChunks, { type: mimeType || 'video/webm' });
      const url = URL.createObjectURL(blob);
      donorCamLive.classList.add('hidden');
      donorCamPlayback.src = url;
      donorCamPlayback.classList.remove('hidden');

      state.donor = {
        type: 'blob',
        file: null,
        blob: blob,
        presetPath: null,
        previewUrl: url,
      };

      updateDonorReadyState(true, `Recorded donor material (${(blob.size / 1024).toFixed(0)} KB)`);
      btnStopDonor.classList.add('hidden');
      btnRetakeDonor.classList.remove('hidden');
    };

    donorRecorder.start(250);
    donorRecordStartTime = Date.now();
    donorRecordIndicator.classList.remove('hidden');
    donorTimerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - donorRecordStartTime) / 1000);
      donorRecordTimer.textContent = formatTime(elapsed);
    }, 500);

    btnRecordDonor.classList.add('hidden');
    btnStopDonor.classList.remove('hidden');
  });

  btnStopDonor.addEventListener('click', () => {
    if (donorRecorder && donorRecorder.state !== 'inactive') {
      donorRecorder.stop();
      clearInterval(donorTimerInterval);
      donorRecordIndicator.classList.add('hidden');
    }
  });

  btnRetakeDonor.addEventListener('click', () => {
    donorCamPlayback.pause();
    donorCamPlayback.classList.add('hidden');
    donorCamLive.classList.remove('hidden');
    btnRetakeDonor.classList.add('hidden');
    btnRecordDonor.classList.remove('hidden');
    updateDonorReadyState(false, 'Ready to record again');
  });

  // --------------------------------------------------------------------------
  // Donor File Upload
  // --------------------------------------------------------------------------
  const dropzoneDonor = document.getElementById('dropzone-donor');
  const fileDonor = document.getElementById('file-donor');
  const previewDonorUpload = document.getElementById('preview-donor-upload');
  const videoDonorPreview = document.getElementById('video-donor-preview');
  const labelDonorFile = document.getElementById('label-donor-file');
  const btnClearDonor = document.getElementById('btn-clear-donor');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneDonor.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneDonor.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneDonor.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneDonor.classList.remove('dragover');
    });
  });

  dropzoneDonor.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleDonorFile(files[0]);
  });

  fileDonor.addEventListener('change', e => {
    if (e.target.files.length > 0) handleDonorFile(e.target.files[0]);
  });

  function handleDonorFile(file) {
    const url = URL.createObjectURL(file);
    videoDonorPreview.src = url;
    labelDonorFile.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    dropzoneDonor.classList.add('hidden');
    previewDonorUpload.classList.remove('hidden');

    state.donor = {
      type: 'file',
      file: file,
      blob: null,
      presetPath: null,
      previewUrl: url,
    };
    updateDonorReadyState(true, `Uploaded Donor: ${file.name}`);
  }

  btnClearDonor.addEventListener('click', () => {
    fileDonor.value = '';
    videoDonorPreview.pause();
    videoDonorPreview.src = '';
    previewDonorUpload.classList.add('hidden');
    dropzoneDonor.classList.remove('hidden');
    state.donor = { type: null, file: null, blob: null, presetPath: null, previewUrl: null };
    updateDonorReadyState(false, 'No donor material selected');
  });

  // --------------------------------------------------------------------------
  // Update Ready States & Trigger Button
  // --------------------------------------------------------------------------
  const statusFaceReady = document.getElementById('status-face-ready');
  const statusDonorReady = document.getElementById('status-donor-ready');
  const cardOriginalFace = document.getElementById('card-original-face');
  const cardDonorMaterial = document.getElementById('card-donor-material');

  function updateFaceReadyState(isReady, text) {
    const textSpan = statusFaceReady.querySelector('.status-text');
    textSpan.textContent = text;
    if (isReady) {
      statusFaceReady.classList.add('ready');
      cardOriginalFace.classList.add('ready');
    } else {
      statusFaceReady.classList.remove('ready');
      cardOriginalFace.classList.remove('ready');
    }
    checkGenerateReadiness();
  }

  function updateDonorReadyState(isReady, text) {
    const textSpan = statusDonorReady.querySelector('.status-text');
    textSpan.textContent = text;
    if (isReady) {
      statusDonorReady.classList.add('ready');
      cardDonorMaterial.classList.add('ready');
    } else {
      statusDonorReady.classList.remove('ready');
      cardDonorMaterial.classList.remove('ready');
    }
    checkGenerateReadiness();
  }

  function checkGenerateReadiness() {
    const faceReady = !!(state.face.file || state.face.blob || state.face.presetPath);
    const donorReady = !!(state.donor.file || state.donor.blob || state.donor.presetPath);
    btnGenerate.disabled = !(faceReady && donorReady);
  }

  // --------------------------------------------------------------------------
  // Deepfake Generation Request
  // --------------------------------------------------------------------------
  btnGenerate.addEventListener('click', async () => {
    btnGenerate.disabled = true;
    sectionProgress.classList.remove('hidden');
    sectionResults.classList.add('hidden');
    sectionProgress.scrollIntoView({ behavior: 'smooth' });

    // Step Simulation
    const steps = [
      document.getElementById('p-step-1'),
      document.getElementById('p-step-2'),
      document.getElementById('p-step-3'),
      document.getElementById('p-step-4'),
    ];

    function activateStep(idx, msg) {
      steps.forEach((s, i) => {
        if (i < idx) {
          s.classList.remove('active');
          s.classList.add('completed');
        } else if (i === idx) {
          s.classList.add('active');
          s.classList.remove('completed');
        } else {
          s.classList.remove('active', 'completed');
        }
      });
      progressStatusText.textContent = msg;
    }

    activateStep(0, 'Normalizing video streams and extracting audio waveforms...');

    const formData = new FormData();

    // Attach Face Material
    if (state.face.file) {
      formData.append('face_file', state.face.file);
    } else if (state.face.blob) {
      formData.append('face_file', state.face.blob, 'face_record.webm');
    } else if (state.face.presetPath) {
      formData.append('face_preset', state.face.presetPath);
    }

    // Attach Donor Material
    if (state.donor.file) {
      formData.append('donor_file', state.donor.file);
    } else if (state.donor.blob) {
      formData.append('donor_file', state.donor.blob, 'donor_record.webm');
    } else if (state.donor.presetPath) {
      formData.append('donor_preset', state.donor.presetPath);
    }

    // Attach Mode & Options
    formData.append('mode', state.mode);
    const tuningRes = document.getElementById('tuning-resolution');
    const tuningPad = document.getElementById('tuning-pad');
    const tuningNosmooth = document.getElementById('tuning-nosmooth');

    formData.append('resize_factor', tuningRes.value);
    formData.append('pad_bottom', tuningPad.value);
    formData.append('nosmooth', tuningNosmooth.checked);

    // Timed step progression for user feedback
    const stepTimer1 = setTimeout(() => activateStep(1, 'Running S3FD Face Detection & 80-band Mel-Spectrogram...'), 2500);
    const stepTimer2 = setTimeout(() => activateStep(2, 'Executing generative synthesis (Wav2Lip GAN / Stream Mux)...'), 6000);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
      });

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      activateStep(3, 'Muxing final video and updating sandbox telemetry...');

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Deepfake generation failed');
      }

      state.generatedResult = data;
      setTimeout(() => displayResults(data), 600);

    } catch (err) {
      alert(`Synthesis Error: ${err.message}`);
      sectionProgress.classList.add('hidden');
      btnGenerate.disabled = false;
    }
  });

  // --------------------------------------------------------------------------
  // Display Results Showcase
  // --------------------------------------------------------------------------
  function displayResults(data) {
    sectionProgress.classList.add('hidden');
    sectionResults.classList.remove('hidden');
    sectionResults.scrollIntoView({ behavior: 'smooth' });

    // Populate Video Showcase Players
    resVideoFace.src = state.face.previewUrl || '';
    resVideoDonor.src = state.donor.previewUrl || '';
    resVideoFake.src = data.output_url;

    // Showcase Footers
    const faceFps = document.getElementById('badge-face-fps');
    if (data.face_info && data.face_info.fps) {
      faceFps.textContent = `${data.face_info.fps} fps`;
    }

    // Populate Telemetry
    telemetryModeTag.textContent = data.mode_title || data.mode;
    tTime.textContent = `${data.generation_time_sec}s`;
    tFilename.textContent = data.filename;
    tPath.textContent = data.output_path;

    if (data.metadata) {
      const m = data.metadata;
      tVideoStream.textContent = `${m.video_codec || 'h264'} (${m.width}x${m.height} @ ${m.fps}fps)`;
      tAudioStream.textContent = `${m.audio_codec || 'aac'} (${m.sample_rate}Hz / ${m.channels}ch)`;
      tSize.textContent = `${m.size_kb} KB`;
    }

    // Download Button
    btnDownloadResult.href = data.output_url;
    btnDownloadResult.setAttribute('download', data.filename);

    btnGenerate.disabled = false;
  }

  // --------------------------------------------------------------------------
  // Synchronized Multi-Player Playback
  // --------------------------------------------------------------------------
  btnSyncPlay.addEventListener('click', () => {
    resVideoFace.currentTime = 0;
    resVideoDonor.currentTime = 0;
    resVideoFake.currentTime = 0;

    resVideoFace.play();
    resVideoDonor.play();
    resVideoFake.play();
  });

  btnSyncPause.addEventListener('click', () => {
    resVideoFace.pause();
    resVideoDonor.pause();
    resVideoFake.pause();
  });

  btnSyncRestart.addEventListener('click', () => {
    resVideoFace.currentTime = 0;
    resVideoDonor.currentTime = 0;
    resVideoFake.currentTime = 0;
    resVideoFace.play();
    resVideoDonor.play();
    resVideoFake.play();
  });

  resVideoFake.addEventListener('timeupdate', () => {
    const cur = resVideoFake.currentTime;
    const dur = resVideoFake.duration || 0;
    masterTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
  });

  // --------------------------------------------------------------------------
  // DeepSentinel Verification Integration
  // --------------------------------------------------------------------------
  btnTestDetector.addEventListener('click', async () => {
    if (!state.generatedResult || !state.generatedResult.filename) return;

    detectorCard.classList.remove('hidden');
    detectorVerdictBadge.textContent = 'ANALYZING...';
    detectorVerdictBadge.className = 'badge badge-accent';
    detectorLiveMsg.innerHTML = '<span class="status-dot"></span> Querying DeepSentinel detector pipeline on port 8000...';

    detectorCard.scrollIntoView({ behavior: 'smooth' });

    try {
      const formData = new FormData();
      formData.append('filename', state.generatedResult.filename);

      const resp = await fetch('/api/send_to_detector', {
        method: 'POST',
        body: formData,
      });

      const res = await resp.json();

      if (!res.detector_active) {
        detectorVerdictBadge.textContent = 'STANDALONE READY';
        detectorVerdictBadge.className = 'badge badge-neutral';
        detectorLiveMsg.innerHTML = `
          <div style="margin-bottom: 0.5rem; color: #f59e0b;">
            ⚠️ <strong>DeepSentinel Detector Not Currently Running on Port 8000</strong>
          </div>
          <p style="font-size: 0.85rem; line-height: 1.5; color: var(--text-secondary);">
            To cross-examine this synthesized deepfake against the defense model live:
            <br>
            1. Open a terminal in this workspace and run: <code>.venv\\Scripts\\python.exe -m webapp</code>
            <br>
            2. Once the detector starts on <code>http://localhost:8000</code>, click this button again or drag <code>generator/outputs/${state.generatedResult.filename}</code> directly into DeepSentinel!
          </p>
        `;
        return;
      }

      // Detector is active and returned results!
      const isFake = res.deepfake_detected;
      const conf = (res.confidence * 100).toFixed(1);

      if (isFake) {
        detectorVerdictBadge.textContent = `DEEPFAKE DETECTED (${conf}%)`;
        detectorVerdictBadge.className = 'badge badge-danger';
      } else {
        detectorVerdictBadge.textContent = `PREDICTED AUTHENTIC (${conf}%)`;
        detectorVerdictBadge.className = 'badge badge-success';
      }

      detectorLiveMsg.innerHTML = `
        <div style="font-size: 0.95rem; line-height: 1.6;">
          <strong>Detector Verdict:</strong> <span style="color: ${isFake ? '#fda4af' : '#6ee7b7'}; font-weight: 700;">${res.label}</span> (${conf}% confidence)
          <br>
          <strong>Manipulation Type:</strong> ${state.generatedResult.mode_title}
          <br>
          <span style="color: var(--text-secondary); font-size: 0.82rem;">The multimodal classifier evaluated emotion incongruence between facial muscle expressions and vocal acoustic prosody.</span>
        </div>
      `;

    } catch (err) {
      detectorVerdictBadge.textContent = 'COMMUNICATION ERROR';
      detectorVerdictBadge.className = 'badge badge-warning';
      detectorLiveMsg.textContent = `Error connecting to detector: ${err.message}`;
    }
  });

  // Reset Studio
  btnResetStudio.addEventListener('click', () => {
    sectionResults.classList.add('hidden');
    sectionProgress.classList.add('hidden');
    detectorCard.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});

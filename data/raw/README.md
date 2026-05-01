# Raw Datasets

The `data/raw/` directory is **excluded from version control** because it contains
gigabytes of audio and video files. Download each dataset manually and place it
at the path shown below before running any pipeline scripts.

---

## CREMA-D (required — Track 1)

**What it is:** Crowd-Sourced Emotional Multimodal Actors Dataset. ~7,400 short
clips of 91 actors (IDs 1001–1091) speaking 12 sentences with 6 emotions
(ANG, DIS, FEA, HAP, NEU, SAD) at varying intensities.

**Expected path:** `data/raw/CREMA-D/`

**How to obtain:**

```bash
# Clone via Git (official repository — includes audio, video, and metadata)
git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D
```

Or download from Kaggle (audio only):
```
https://www.kaggle.com/datasets/ejlok1/cremad
```

**Contents after download:**

| Subdirectory / File | Description |
|---------------------|-------------|
| `AudioWAV/`         | ~7,400 WAV audio clips (one per clip) |
| `VideoFlash/`       | ~7,442 FLV video clips (one per clip) |
| `finishedEmoResponses.csv` | Crowd-sourced emotion labels |
| `VideoDemographics.csv`    | Actor age/gender/race demographics |
| `SentenceFilenames.csv`    | Sentence code ↔ full text mapping |

**Filename format:**

```
1001_IEO_ANG_HI.wav
│    │   │   └── Intensity: HI / MD / LO / XX
│    │   └── Emotion:   ANG DIS FEA HAP NEU SAD
│    └── Sentence code (12 codes: IEO, TIE, IOM, IWW, TAI, MTI, IWL, ITH, DFA, ITS, TSI, WSI)
└── Actor ID: 1001–1091
```

---

## SAVEE (optional — future tracks)

**What it is:** Surrey Audio-Visual Expressed Emotion. 480 clips from 4 male
actors (DC, JE, JK, KL) across 7 emotions.

**Expected path:** `data/raw/SAVEE/`

**How to obtain:**

Request access through the official form:
```
http://kahlan.eps.surrey.ac.uk/savee/
```

Alternatively, a community mirror is available on Kaggle:
```
https://www.kaggle.com/datasets/barelydedicated/savee-database
```

---

## MELD (optional — future tracks)

**What it is:** Multimodal EmotionLines Dataset. ~1,400 dialogues from Friends TV show,
each clip labeled with 7 emotions. Contains audio, video, and text modalities.

**Expected path:** `data/raw/MELD/`

**How to obtain:**

```bash
# Download via the official Zenodo release
# https://zenodo.org/record/3989519
# Or from the GitHub repository:
git clone https://github.com/declare-lab/MELD.git data/raw/MELD
```

**Contents:**
- `MELD-RAW/MELD.Raw/train/`, `dev/`, `test/` — split MP4 video clips
- `MELD-RAW/MELD.Raw/*_sent_emo.csv` — utterance-level sentiment + emotion labels
- `MELD-Features-Models/` — pre-extracted audio/text/video features

---

## CMU-MOSEI (optional — future tracks)

**What it is:** Large-scale multimodal sentiment and emotion dataset.

**Expected path:** `data/raw/CMU-MOSEI/`

**How to obtain:**

```
http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
```

Or via the SDK:
```bash
pip install mmsdk
python -c "import mmsdk; mmsdk.mmdatasdk.computational_sequence.CMU_MOSEI.highlevel"
```

---

## Directory structure after setup

```
data/raw/
├── README.md          ← this file
├── CREMA-D/           ← cloned from GitHub (~3 GB)
│   ├── AudioWAV/
│   ├── VideoFlash/    (or root-level .flv files depending on version)
│   └── ...
├── SAVEE/             ← downloaded separately (~300 MB)
│   └── *.wav
└── CMU-MOSEI/         ← downloaded separately (optional)
```

---

## Disk space requirements

| Dataset  | Approximate size |
|----------|-----------------|
| CREMA-D  | ~3 GB           |
| SAVEE    | ~300 MB         |
| CMU-MOSEI | ~30 GB (full) |

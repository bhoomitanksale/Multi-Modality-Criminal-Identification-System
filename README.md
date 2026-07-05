# Criminal Identification System v2 — Algorithm Reference

> **Team BioFuse | IGNISIA | AI for Public Safety**
> Last updated: March 2026

---

## 1. System Overview

CIS v2 is a **multi-modal biometric identification pipeline** that fuses three independent AI analysis engines to classify a subject as:

| Verdict | Fusion Score |
|---|---|
| 🚨 **CRIMINAL** | > 62% |
| ⚠️ **WATCH LIST** | 45% – 62% |
| ✅ **CLEAR** | < 45% |

The three engines run **in parallel** on every frame and their outputs are blended by a weighted Fusion Engine.

```
Frame Input
    │
    ├──► Face Analyzer    (weight 45%)
    ├──► Gait Analyzer    (weight 30%)
    └──► Behavior Analyzer (weight 25%)
              │
              ▼
         Fusion Engine
              │
              ▼
     Verdict  +  Confidence %
```

---

## 2. Face Analyzer (`modules/face_analyzer.py`)

### 2.1 Detection — Haar Cascade (Viola-Jones)

- **Algorithm**: OpenCV `haarcascade_frontalface_default.xml` (Viola-Jones 2001)
- **How it works**: Scans the image at multiple scales using a cascade of simple Haar-like rectangular feature detectors. Each stage is a boosted classifier (AdaBoost). Only patches that pass *all* stages are classified as faces — making it very fast.
- **Settings used**:
  - `scaleFactor = 1.15` — downsample ratio per pyramid level
  - `minNeighbors = 4` — minimum overlapping detections to confirm a face
  - `minSize = 40×40 px` — smallest accepted face region
- **Fallback**: If no frontal face is found, profile view (`haarcascade_profileface.xml`) is tried.
- **Selection**: When multiple faces are detected, the **largest bounding box** (by area) is used.

### 2.2 Eye Detection

- **Algorithm**: `haarcascade_eye.xml` (Haar cascade for eyes)
- Applied to the **upper half** of the detected face ROI only.
- `minNeighbors = 8` (stricter to reduce false positives)
- Provides:
  - `eye_count` (0, 1, or 2)
  - Normalized **inter-ocular distance** (eye_distance = IOD / face_width)

### 2.3 Feature Extraction

| Feature | Method |
|---|---|
| Brightness | Mean of grayscale face ROI (0–255) |
| Contrast | Standard deviation of grayscale ROI |
| Sharpness | Laplacian variance — measures edge clarity |
| Face symmetry | L2 pixel difference between left and mirrored right half |
| Head tilt | `atan2(Δy, Δx)` of eye center vector (degrees) |
| Face ratio | width / height of bounding box |

### 2.4 32-Dimensional Feature Vector

Built for database similarity comparison:
- Resize face ROI to **8×4 pixels** (= 32 flattened pixels, normalized 0–1)
- Last 4 dimensions replaced with: `[eye_distance, face_ratio, brightness/255, contrast/128]`

### 2.5 Quality Score (0–1)

Penalizes/rewards factors:

| Condition | Score Modifier |
|---|---|
| Face area 4000–80000 px² | +0.15 |
| Face area < 1600 px² | –0.20 |
| Sharpness > 100 | +0.15 |
| Sharpness < 30 | –0.15 |
| 2 eyes visible | +0.20 |
| 1 eye visible | +0.05 |
| Good brightness (75–200) | +0.10 |

Baseline = 0.50. Clipped to [0, 1].

### 2.6 Suspicion Score (0–1)

| Factor | Score Modifier |
|---|---|
| Base | +0.40 |
| Masked / no eyes | +0.22 |
| Eye count = 0 | +0.15 |
| Two eyes (cooperative) | –0.08 |
| Head tilt > 20° | +0.08 |
| Dark/overexposed (< 60 or > 220) | +0.08 |
| Low sharpness < 25 | +0.07 |
| Low face symmetry < 0.4 | +0.06 |
| Random noise (realism) | ±0.04 |

Clipped to [0, 1].

---

## 3. Gait Analyzer (`modules/gait_analyzer.py`)

### 3.1 What It Detects

Gait analysis measures **body motion across frames** — how a person moves, sways, and bounces while walking.

### 3.2 Algorithm

1. **Background subtraction** — `cv2.createBackgroundSubtractorMOG2()`
   - Mixture of Gaussians (MoG2) models the background pixel distribution
   - Foreground (moving) pixels are extracted as the motion mask
2. **Contour extraction** — All foreground contours are found; a bounding box encloses the full moving region (person silhouette)
3. **Silhouette Energy Image (SEI)** — Accumulated binary foreground masks over N frames are summed to produce a motion energy map
4. **Gait features extracted**:
   - **Step frequency** — peaks in vertical centroid oscillation (scipy FFT-based peak detection or simplified version)
   - **Torso sway** — variance in horizontal centroid over frames
   - **Arm swing** — estimated from lateral extent changes per frame
   - **Vertical bounce** — RMS of vertical centroid delta series

### 3.3 Confidence Score

Pattern matching against stored gait templates in the criminal database. Similarity is computed via **cosine similarity** on the gait feature vector. In demo mode without a full database, a baseline behavioral score is returned.

---

## 4. Behavior Analyzer (`modules/behavior_analyzer.py`)

### 4.1 Behavioral Indicators

Detects suspicious **behavioral cues** from frame sequences:

| Behavior | Detection Method |
|---|---|
| Loitering | Subject remains in the same position for > N frames |
| Erratic movement | High variance in displacement vectors |
| Proximity to restricted zones | ROI overlap with configured zones |
| Crowd density | Number of detected people-like contours |
| Concealment posture | Unusual bounding box shape (extreme aspect ratios) |

### 4.2 Algorithm

1. **Optical flow** — `cv2.calcOpticalFlowPyrLK()` (Lucas-Kanade pyramidal) — tracks point features between consecutive frames to measure motion vectors
2. **Motion vector statistics** — mean magnitude, directional consistency, and temporal variance are derived from the flow field
3. **Face region exclusion** — if a face bounding box is provided, its pixels are excluded from the motion field to isolate body behavior
4. **Temporal windowing** — a sliding buffer of N=15 frames is maintained; statistics are computed over the window

### 4.3 Confidence Score

Weighted sum of behavioral indicator activations, clipped to [0, 1]. Randomness ±3% added for realism in demo mode.

---

## 5. Fusion Engine (`modules/fusion_engine.py`)

### 5.1 Adaptive Weighted Fusion

The three module scores are blended using **adaptive weights**:

| Module | Default Weight | Symbol |
|---|---|---|
| Face Recognition | 45% | `w_f` |
| Gait Analysis | 30% | `w_g` |
| Behavior Analysis | 25% | `w_b` |

Weights are **dynamically adjusted** based on availability:
- If a module did not detect its target (e.g., no face found), its weight is redistributed proportionally to the other active modules.
- This prevents a failed module from dragging down the fusion score.

### 5.2 Consensus Bonus

If **two or more modules agree** (both score above their individual thresholds):
- Each additional agreeing module adds `+0.08` to the fusion score (configurable: `CONSENSUS_BONUS = 0.08`)
- This rewards convergent evidence

### 5.3 Multi-Scan Aggregation (Webcam Mode)

For webcam scans, **3 full scan passes** are performed. After all passes:
- Each scan's result is stored
- Final fusion = **weighted average** of all scan passes, giving higher weight to scans with higher individual confidence
- This produces a much more reliable verdict than a single snapshot

### 5.4 Database Similarity

The 32-dim face feature vector is compared against all criminal records in the SQLite database using:

- **Cosine Similarity**: `sim = (A·B) / (|A| × |B|)`
- The record with the highest similarity (if > threshold) is returned as the suspect match
- Match info: suspect name, alias, crime type, risk level

### 5.5 Verdict Assignment

```
fusion_score = Σ (w_i × conf_i)   [adjusted + consensus bonus]

if fusion_score > 0.62  →  CRIMINAL
elif fusion_score > 0.45 →  WATCH LIST
else                    →  CLEAR
```

---

## 6. 3-Scan Webcam Pipeline (`modules/scanner.py`)

```
Start
  │
  ├─ Open camera (DSHOW backend → ANY backend)
  │     ↑ MJPEG codec hint + 10-frame warmup retry
  │
  ├─ SCAN 1: capture 30 frames, run all 3 modules, fuse
  │     └─ 1.2 s pause (live preview continues)
  │
  ├─ SCAN 2: capture 30 frames, run all 3 modules, fuse
  │     └─ 1.2 s pause
  │
  ├─ SCAN 3: capture 30 frames, run all 3 modules, fuse
  │
  └─ Aggregate all 3 scan results → FINAL VERDICT
```

**Key design decisions**:
- Best-quality face per scan pass is selected (highest quality score) to avoid a blurry frame corrupting the result
- Between-scan frames are still displayed live so the UI never freezes

---

## 7. Camera System (Fixed in v2.1)

### 7.1 Auto-Detection Logic

On Windows (most laptop webcam failures happen here):

1. Try `cv2.VideoCapture(index, cv2.CAP_DSHOW)` first — DirectShow is the native Windows webcam API
2. Fall back to `cv2.CAP_ANY` (auto-select backend)
3. Set MJPEG fourcc codec hint for faster initialization
4. Retry frame read **up to 10 times** (some cameras take 300–500ms to produce a valid first frame)
5. Auto-scan indices 0–4 if `Auto` mode is selected

### 7.2 Common Camera Fix Checklist

| Problem | Fix |
|---|---|
| Camera busy (Teams/Zoom open) | Close all video apps first |
| Wrong index | Use "Test Cam" button to probe each index |
| Driver issue | Device Manager → Update camera driver |
| Disabled by privacy setting | Windows Settings → Privacy → Camera → Allow apps |
| Fn key disabled | Press Fn + F7 (or relevant key) to re-enable |

---

## 8. Database (`database/db_manager.py`)

- **Engine**: SQLite (no server needed, file-based)
- **Tables**:
  - `criminals` — registry of known suspects (name, alias, crime type, risk level, face vector)
  - `detections` — log of every scan event with timestamp, verdict, mode, confidence scores
- **Operations**: `init_db()`, `get_all_criminals()`, `log_detection_event()`, `get_stats()`, `get_recent_detections()`

---

## 9. Performance Notes

| Setting | Value | Effect |
|---|---|---|
| `VIDEO_SAMPLE_EVERY_N = 5` | Analyze every 5th frame | Speeds up video processing 5× |
| `WEBCAM_SCAN_FRAMES = 30` | 30 frames per scan | ~1 second per scan at 30fps |
| `SCAN_INTERVAL_SEC = 1.2` | Pause between scans | Reduces false positives from motion blur |
| `FACE_MIN_SIZE = 40×40` | Minimum face size | Reduces false detections from tiny regions |

---

## 10. Module File Map

```
criminal_identification_system_v2/
├── cis_v2.py                  ← Main GUI application (4-tab Tkinter)
├── modules/
│   ├── face_analyzer.py       ← Haar cascade face + eye detection, feature extraction
│   ├── gait_analyzer.py       ← MOG2 background subtraction, silhouette energy, gait features
│   ├── behavior_analyzer.py   ← Lucas-Kanade optical flow, loitering, erratic motion
│   ├── fusion_engine.py       ← Adaptive weighted fusion, cosine DB similarity, multi-scan aggregation
│   └── scanner.py             ← Pipeline orchestrator, camera open logic, 3-scan cycle
├── database/
│   └── db_manager.py          ← SQLite CRUD, detection logging
├── config/
│   └── settings.py            ← All thresholds, weights, paths
└── ALGORITHM_README.md        ← This file
```

---

*CIS v2 — BioFuse Team | IGNISIA Competition*

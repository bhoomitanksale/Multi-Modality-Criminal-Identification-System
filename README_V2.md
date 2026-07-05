# CIS v2 — Criminal Identification System (Enhanced)
### BioFuse Team | IGNISIA | AI for Public Safety

---

## 🚀 Quick Start

**Double-click `START_V2.bat`** — or run:
```bash
cd criminal_identification_system_v2
python cis_v2.py
```

---

## 📋 System Requirements

| Requirement | Version |
|-------------|---------|
| Python      | 3.8+    |
| opencv-python | 4.8+  |
| pillow      | 10.0+   |
| numpy       | 1.24+   |
| mediapipe *(optional)* | 0.10+ |

Install deps:
```bash
pip install opencv-python pillow numpy
# For gait analysis:
pip install mediapipe
```

---

## 🖥️ Application Tabs

### Tab 1 — 📷 Image Upload
- Browse and load any image (`.jpg`, `.png`, `.bmp`, `.webp`)
- Runs face detection + behavioral analysis
- Shows annotated image with bounding box + verdict
- Confidence breakdown: Face / Gait / Behavior / Fusion

### Tab 2 — 📹 Video Upload
- Load `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- Frame-by-frame analysis with live overlay
- Progress bar + real-time verdict updates
- Final aggregated report after full video

### Tab 3 — 🎥 Live Webcam (3-Scan Pipeline)
```
Scan 1/3 ──► Face primary analysis (30 frames)
     ↓
Scan 2/3 ──► Gait + behavioral analysis (30 frames)
     ↓
Scan 3/3 ──► Final consolidation + DB lookup (30 frames)
     ↓
Final Verdict: CRIMINAL | WATCH LIST | CLEAR
```

### Tab 4 — 🗄️ Database
- View all 10 pre-seeded criminal suspects
- Search/filter by name, alias, or crime type
- Detection events log with timestamps
- Stats: total scans, criminal hits, watch list, cleared

---

## 🔬 Detection Algorithm

### Multi-Modal Fusion
| Modality | Default Weight | Detects |
|----------|---------------|---------|
| Face Recognition | 45% | Eye contact, symmetry, masking, gaze |
| Gait Analysis | 30% | Arm asymmetry, stride, torso stability |
| Behavioral | 25% | Loitering, avoidance, head jitter |

### Adaptive Weight Redistribution
If a modality is unavailable (face hidden, no gait data), its weight is redistributed to active modalities proportionally.

### Verdict Thresholds
| Verdict | Fusion Score |
|---------|-------------|
| 🚨 CRIMINAL | ≥ 62% |
| ⚠️ WATCH LIST | 45% – 61% |
| ✅ CLEAR | < 45% |

### Consensus Bonus
+8% confidence when 2+ modalities agree on high suspicion (≥55% each).

---

## 🗄️ Database Schema

### `criminals` table
- 10 pre-seeded suspects with names, aliases, crime types, risk levels
- Each has a 32-dimensional face biometric feature vector

### `face_biometrics` table
- Cosine similarity search against stored feature vectors
- Match > 70% → +18% confidence boost
- Match > 55% → +8% confidence boost

### `detection_events` table
- Logs every scan with timestamp, mode, confidence scores, and verdict

---

## 📁 Project Structure

```
criminal_identification_system_v2/
├── cis_v2.py               ← Main GUI application (run this)
├── START_V2.bat            ← Windows launcher
├── requirements.txt
├── README_V2.md
│
├── config/
│   └── settings.py         ← Thresholds, weights, UI colours
│
├── database/
│   ├── db_manager.py       ← SQLite CRUD + cosine similarity search
│   └── cis_v2.sqlite       ← Auto-created on first run
│
├── modules/
│   ├── face_analyzer.py    ← Haar cascade + 32-dim feature vectors
│   ├── gait_analyzer.py    ← MediaPipe Pose + fallback heuristics
│   ├── behavior_analyzer.py← Optical flow + gaze + trajectory
│   ├── fusion_engine.py    ← Adaptive weighted fusion + DB lookup
│   └── scanner.py          ← 3-scan orchestrator + callbacks
│
└── logs/                   ← Log files (auto-created)
```

---

## ⚠️ Legal Disclaimer

> This system is for **authorized law enforcement and security research purposes only**.  
> Unauthorized surveillance or biometric collection may violate applicable laws.  
> All data is stored locally — no cloud transmission.

---

*CIS v2 | BioFuse | IGNISIA 2026*

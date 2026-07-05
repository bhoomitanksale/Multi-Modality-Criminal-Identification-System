"""
CIS v2 - Known Criminals High-Priority Matcher
===============================================
Uses OpenCV LBPH face recognizer trained on reference images stored in
  data/criminal_faces/<criminal_id>/*.jpg

Workflow:
  1. On startup, scan data/criminal_faces/ for sub-folders
  2. Train LBPH recognizer on any face images found there
  3. Expose match_face(face_roi_bgr) -> HighPriorityMatch | None

Criminal metadata (from open-source records) is hardcoded below.
Reference images must be placed manually by the user in:
  data/criminal_faces/
    dawood_ibrahim/   ← put any photo of Dawood here (jpg/png)
    osama_bin_laden/
    ... etc.

The system tries LBPH first; if no reference images after reading the
directory, it still provides instant metadata if the DB has a face match.
"""

import cv2
import numpy as np
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "data", "criminal_faces")

# ─── Hardcoded Criminal Metadata ─────────────────────────────────────────────
# Sources: INTERPOL, FBI Most Wanted, public court records
CRIMINAL_METADATA: dict[str, dict] = {}


# ─── Result Dataclass ─────────────────────────────────────────────────────────
@dataclass
class HighPriorityMatch:
    criminal_id:  str
    name:         str
    alias:        str
    crime_type:   str
    risk_level:   str
    status:       str
    known_for:    str
    bounty:       str
    priority:     str          # 'CRITICAL' or 'HIGH'
    confidence:   float = 0.0  # 0–1, how confident the match is
    source:       str  = "lbph" # 'lbph' or 'metadata_only'


# ─── Matcher ─────────────────────────────────────────────────────────────────
class KnownCriminalMatcher:
    """
    Loads reference images from data/criminal_faces/,
    trains OpenCV LBPH face recognizer,
    and provides real-time face matching.
    """

    def __init__(self):
        self._recognizer    = None
        self._label_map:    dict[int, str] = {}   # label → criminal_id
        self._trained       = False
        self._face_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._train()

    # ── Public API ────────────────────────────────────────────────────────────
    def match_face(self, face_roi_bgr: np.ndarray) -> Optional[HighPriorityMatch]:
        """
        Try to match the supplied face ROI against trained criminals.
        Returns HighPriorityMatch on confident match, else None.
        face_roi_bgr: BGR numpy array of cropped face region.
        """
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return None
        if not self._trained or self._recognizer is None:
            return None

        try:
            gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            # equalizeHist improves lighting invariance
            gray = cv2.equalizeHist(gray)

            label, dist = self._recognizer.predict(gray)
            # LBPH dist: lower = better. Typical threshold: < 80 good, < 110 ok
            if dist > 120.0:
                return None

            criminal_id = self._label_map.get(label)
            if not criminal_id:
                return None

            meta = CRIMINAL_METADATA.get(criminal_id, {})
            confidence = float(np.clip(1.0 - (dist / 120.0), 0.0, 1.0))

            return HighPriorityMatch(
                criminal_id = criminal_id,
                name        = meta.get("name", criminal_id),
                alias       = meta.get("alias", ""),
                crime_type  = meta.get("crime_type", "Unknown"),
                risk_level  = meta.get("risk_level", "HIGH"),
                status      = meta.get("status", "Unknown"),
                known_for   = meta.get("known_for", ""),
                bounty      = meta.get("bounty", ""),
                priority    = meta.get("priority", "HIGH"),
                confidence  = confidence,
                source      = "lbph",
            )

        except Exception as e:
            print(f"[KnownCriminalMatcher] match error: {e}")
            return None

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def trained_count(self) -> int:
        return len(self._label_map)

    # ── Training ──────────────────────────────────────────────────────────────
    def _train(self):
        """Scan data/criminal_faces/, extract face chips, train LBPH."""
        if not os.path.isdir(FACES_DIR):
            os.makedirs(FACES_DIR, exist_ok=True)
            self._create_placeholder_readme()
            return

        images, labels = [], []
        label_counter  = 0

        for criminal_id in os.listdir(FACES_DIR):
            folder = os.path.join(FACES_DIR, criminal_id)
            if not os.path.isdir(folder):
                continue

            face_list = self._load_faces_from_folder(folder)
            if not face_list:
                continue

            self._label_map[label_counter] = criminal_id
            for face_chip in face_list:
                images.append(face_chip)
                labels.append(label_counter)
            print(f"[LBPH] Loaded {len(face_list)} face(s) for: {criminal_id}")
            label_counter += 1

        if images:
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=200.0
            )
            recognizer.train(images, np.array(labels))
            self._recognizer = recognizer
            self._trained    = True
            print(f"[LBPH] Trained on {len(images)} images for {label_counter} criminals.")
        else:
            print("[LBPH] No reference images found in data/criminal_faces/. "
                  "Add sub-folders with criminal photos to enable face matching.")

    def _load_faces_from_folder(self, folder: str) -> list:
        """Load and chip faces from all images in the folder."""
        chips = []
        exts  = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for fname in os.listdir(folder):
            if not fname.lower().endswith(exts):
                continue
            path = os.path.join(folder, fname)
            try:
                img  = cv2.imread(path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
                    chip = gray[y:y+h, x:x+w]
                elif gray.shape[0] > 30 and gray.shape[1] > 30:
                    # Use the entire image as face chip if no face detected
                    chip = gray
                else:
                    continue

                chip = cv2.resize(chip, (100, 100))
                chip = cv2.equalizeHist(chip)
                chips.append(chip)
            except Exception as e:
                print(f"[LBPH] Error loading {fname}: {e}")
        return chips

    def _create_placeholder_readme(self):
        """Create a README in the criminal faces directory."""
        readme = os.path.join(FACES_DIR, "README.txt")
        with open(readme, "w") as f:
            f.write("""CIS v2 - Criminal Reference Faces
===================================
Create a sub-folder for each criminal using their ID (see list below).
Place any JPG/PNG photos of that person in the folder.
The system will auto-train LBPH face recognizer on restart.

Supported criminal IDs:
""")
            for cid, meta in CRIMINAL_METADATA.items():
                f.write(f"  {cid:<30} → {meta['name']}\n")
            f.write("""
Example:
  data/criminal_faces/
    dawood_ibrahim/
      dawood1.jpg
      dawood2.jpg
    osama_bin_laden/
      osama1.jpg
""")

# ─── Singleton instance ───────────────────────────────────────────────────────
_matcher: Optional[KnownCriminalMatcher] = None

def get_matcher() -> KnownCriminalMatcher:
    """Return the singleton KnownCriminalMatcher, initializing if needed."""
    global _matcher
    if _matcher is None:
        _matcher = KnownCriminalMatcher()
    return _matcher

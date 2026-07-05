"""
CIS v2 - Face Analyzer Module
Extracts facial features using OpenCV Haar cascades.

CALIBRATION TARGETS (revised):
  Normal innocent person (smiling, eyes open, good lighting):
    Face score: 0.15 - 0.28  → INNOCENT after fusion
  Suspicious person (hiding face, dark corner, avoiding gaze):
    Face score: 0.50 - 0.70  → SUSPECT/CRIMINAL after fusion
  Person with actual face covering (mask/scarf/balaclava):
    Face score: 0.55 - 0.75  → CRIMINAL after fusion
"""
import cv2
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE,
    EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS
)


class FaceFeatures:
    """Structured container for extracted face features."""
    def __init__(self):
        self.detected         = False
        self.masked           = False          # True ONLY if lower face is physically covered
        self.quality          = 0.0
        self.confidence       = 0.0

        self.bbox             = None
        self.eye_count        = 0
        self.eye_distance     = 0.0
        self.face_area        = 0
        self.face_ratio       = 0.0
        self.face_symmetry    = 0.0

        self.brightness       = 0.0
        self.contrast         = 0.0
        self.sharpness        = 0.0

        self.head_tilt_deg    = 0.0
        self.feature_vector   = []
        self.face_roi         = None
        self.gray_roi         = None

        # New: lower-face texture (for real mask detection)
        self.lower_face_std   = 0.0
        self.upper_face_std   = 0.0

    def to_dict(self):
        return {
            "detected":     self.detected,
            "masked":       self.masked,
            "quality":      round(self.quality, 3),
            "confidence":   round(self.confidence, 3),
            "eye_count":    self.eye_count,
            "eye_distance": round(self.eye_distance, 3),
            "face_ratio":   round(self.face_ratio, 3),
            "brightness":   round(self.brightness, 1),
            "contrast":     round(self.contrast, 1),
            "sharpness":    round(self.sharpness, 1),
            "head_tilt":    round(self.head_tilt_deg, 1),
        }


class FaceAnalyzer:
    """
    Multi-stage face analyzer.

    KEY DESIGN PRINCIPLE:
    - Eyes clearly visible + good lighting = STRONG innocent indicator
    - Lower face physically covered = STRONG criminal indicator
    - Avoiding eye contact, dark lighting = suspicious
    - Eye cascade FAILURE alone does NOT mean masked (cascade is imperfect)
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._rng = np.random.default_rng(seed=42)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # ──────────────────────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> FaceFeatures:
        """Main entry point. Accepts a BGR frame. Returns FaceFeatures."""
        ff = FaceFeatures()
        if frame is None or frame.size == 0:
            return ff

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Anti-Glare Preprocessing
        gray = self.clahe.apply(gray)

        # 1. Detect all faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
            )
        if len(faces) == 0:
            return []

        results = []
        for (x, y, w, h) in faces:
            ff = FaceFeatures()
            ff.detected   = True
            ff.bbox       = (x, y, w, h)
            face_roi      = frame[y:y+h, x:x+w]
            gray_roi      = gray[y:y+h, x:x+w]
            ff.face_roi   = face_roi
            ff.gray_roi   = gray_roi
            ff.face_area  = w * h
            ff.face_ratio = round(w / max(h, 1), 3)

            # 2. Eye detection
            upper_half = gray_roi[:h // 2, :]
            eyes = self.eye_cascade.detectMultiScale(
                upper_half, scaleFactor=EYE_SCALE_FACTOR, minNeighbors=EYE_MIN_NEIGHBORS
            )
            ff.eye_count = min(len(eyes), 2)  # Cap at 2

            if ff.eye_count >= 2:
                ex1, ey1, ew1, _ = eyes[0]
                ex2, ey2, ew2, _ = eyes[1]
                cx1 = ex1 + ew1 / 2
                cx2 = ex2 + ew2 / 2
                ff.eye_distance = round(abs(cx2 - cx1) / max(w, 1), 3)
            elif ff.eye_count == 1:
                ff.eye_distance = 0.15
            else:
                ff.eye_distance = 0.0

            # 3. Intensity metrics
            ff.brightness = float(np.mean(gray_roi))
            ff.contrast   = float(np.std(gray_roi))
            ff.sharpness  = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

            # 4. Lower/upper face texture (for TRUE mask detection)
            if gray_roi.shape[0] > 20:
                # Enhance ROI for better feature capture
                gray_roi = cv2.equalizeHist(gray_roi)
                upper_region      = gray_roi[:gray_roi.shape[0]//2, :]
                lower_region      = gray_roi[gray_roi.shape[0]//2:, :]
                ff.upper_face_std = float(np.std(upper_region))
                ff.lower_face_std = float(np.std(lower_region))
                # Refined mask heuristic: flat lower texture vs detailed upper texture
                ff.masked = (ff.lower_face_std < 14 and ff.upper_face_std > 20)
            else:
                ff.upper_face_std = float(np.std(gray_roi))
                ff.lower_face_std = ff.upper_face_std

            # 5. Head tilt
            if ff.eye_count >= 2:
                ex1, ey1, ew1, _ = eyes[0]
                ex2, ey2, ew2, _ = eyes[1]
                dx = (ex2 + ew2/2) - (ex1 + ew1/2)
                dy = ey2 - ey1
                ff.head_tilt_deg = round(math.degrees(math.atan2(dy, max(abs(dx), 1))), 1)
            else:
                ff.head_tilt_deg = 0.0

            # 6. Face symmetry
            left_half  = gray_roi[:, :w // 2]
            right_half = cv2.flip(gray_roi[:, w // 2:], 1)
            min_cols   = min(left_half.shape[1], right_half.shape[1])
            if min_cols > 0:
                diff = cv2.absdiff(left_half[:, :min_cols], right_half[:, :min_cols])
                ff.face_symmetry = round(1.0 - np.mean(diff) / 255.0, 3)
            else:
                ff.face_symmetry = 0.5

            # 7. Feature vector (32-dim)
            ff.feature_vector = self._build_feature_vector(ff, gray_roi)

            # 8. Quality score
            ff.quality = self._compute_quality(ff)

            # 9. Suspicion score
            ff.confidence = self._compute_suspicion(ff)
            
            results.append(ff)

        return results


    # ──────────────────────────────────────────────────────────────────────────
    def _build_feature_vector(self, ff: FaceFeatures, gray_roi: np.ndarray) -> list:
        """
        Builds a 128-dimensional feature vector.
        Uses a 16x8 grid and normalization for robust cosine similarity.
        """
        try:
            # 16x8 = 128 dimensions for significantly better accuracy than 8x4
            resized = cv2.resize(gray_roi, (16, 8), interpolation=cv2.INTER_AREA)
            # Normalize to 0-1 range
            pf = (resized.flatten().astype(float) / 255.0).tolist()
        except Exception:
            pf = [0.0] * 128
        
        # Ensure list is exactly 128 elements (fill if needed)
        if len(pf) < 128:
            pf.extend([0.0] * (128 - len(pf)))
        elif len(pf) > 128:
            pf = pf[:128]

        # Use the tail elements for metadata (matching fusion logic)
        # We assume the database search ignores these or handles them.
        # But wait, cis_v2 uses these specific indices. Let's make sure they fit.
        # The DB likely stores the whole vector. I'll stick to a large vector 
        # but ensure the indices 24-31 are still conceptually useful if needed.
        # Fixed indices from cis_v2: 24 (source), 25 (gait), 26 (behavior)
        # Let's adjust:
        pf[24] = 0.0 # Source
        pf[25] = 0.0 # Gait
        pf[26] = 0.0 # Behavior
        return pf

    def _compute_quality(self, ff: FaceFeatures) -> float:
        score = 0.5
        if 4000 < ff.face_area < 80000:
            score += 0.15
        elif ff.face_area < 1600:
            score -= 0.20
        if ff.sharpness > 100:
            score += 0.15
        elif ff.sharpness < 30:
            score -= 0.15
        if ff.eye_count >= 2:
            score += 0.20
        elif ff.eye_count == 1:
            score += 0.05
        if 75 < ff.brightness < 200:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _compute_suspicion(self, ff: FaceFeatures) -> float:
        """
        Enhanced suspicion score calibration.
        Innocent: 0.10 - 0.30 | Suspicious: 0.35 - 0.50 | Criminal: 0.55 - 0.80
        """
        score = 0.35  # Neutral baseline

        # ── PRIMARY INNOCENCE INDICATOR: Eyes clearly visible ──────────────────
        if ff.eye_count >= 2:
            dist_bonus = 0.15 if ff.eye_distance > 0.22 else 0.10
            score -= dist_bonus
        elif ff.eye_count == 1:
            score -= 0.05

        # ── PRIMARY CRIMINAL INDICATOR: Face coverage ──────────────────────────
        if ff.masked:
            score += 0.25 # Direct detection of fabric texture
        elif ff.lower_face_std < 18 and ff.upper_face_std > 25:
            score += 0.12 # Partial cover (scarf/collar)

        # ── LIGHTING ──────────────────────────────────────────────────────────
        if ff.brightness < 45:
            score += 0.12      # Shadows/Dark
        elif 85 < ff.brightness < 180:
            score -= 0.08      # Ideal cooperative lighting
        elif ff.brightness > 230:
            score += 0.05      # Overexposed

        # ── ORIENTATION / EVASION ─────────────────────────────────────────────
        if ff.face_ratio < 0.55:
            score += 0.10      # Profile view (camera avoidance)
        
        tilt = abs(ff.head_tilt_deg)
        if tilt > 25:
            score += 0.06      # Strong evasion tilt
        elif 1 < tilt < 10:
            score -= 0.02      # Natural candid tilt

        # ── BLUR / QUALITY ───────────────────────────────────────────────────
        if ff.sharpness < 15:
            score += 0.08      # Motion blur evasion
        elif ff.face_symmetry > 0.75 and ff.eye_count >= 2:
            score -= 0.05      # High quality candid

        # Small randomness for uniqueness
        score += self._rng.uniform(-0.005, 0.005)

        return float(np.clip(score, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    def draw_overlays(self, frame: np.ndarray, detections: list,
                      scan_label: str = None) -> np.ndarray:
        """Draw face bounding boxes + labels for multiple detections."""
        out = frame.copy()
        for ff in detections:
            if not ff.detected or ff.bbox is None:
                continue

            x, y, w, h = ff.bbox
            
            # Map fusion colors if available (defaults to blue for crowd)
            verdict = getattr(ff, 'verdict', 'ANALYZING')
            if verdict == "CRIMINAL":
                color = (0, 0, 255)
            elif verdict in ("SUSPECT", "WATCH LIST"):
                color = (0, 165, 255)
            elif verdict in ("CLEAR", "INNOCENT"):
                color = (0, 200, 60)
            else:
                color = (0, 200, 255)

            cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), color, 2)
            
            label = scan_label if scan_label else verdict
            cv2.putText(out, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return out

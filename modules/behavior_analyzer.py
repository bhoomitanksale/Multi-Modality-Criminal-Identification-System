"""
CIS v2 - Behavioral Analyzer Module
Detects suspicious behavioral patterns from video frames:
- Gaze avoidance
- Nervous micro-movements (head jitter, shoulder tension)
- Trajectory patterns (loitering, erratic movement)
- Body orientation (camera avoidance)
"""
import cv2
import numpy as np
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BehaviorFeatures:
    """Container for behavioral analysis results."""
    def __init__(self):
        self.detected           = False
        self.confidence         = 0.0   # Suspicion score 0–1

        # Gaze
        self.gaze_away          = False  # Not looking at camera
        self.gaze_evasiveness   = 0.0   # 0–1

        # Movement
        self.head_jitter        = 0.0   # Frame-to-frame head movement variance
        self.body_tension       = 0.0   # Shoulder elevation relative to neck
        self.movement_speed     = 0.0   # Average movement across frames

        # Trajectory
        self.trajectory_erratic = False
        self.loitering_score    = 0.0   # High = staying in one place (suspicious)
        self.path_straightness  = 1.0   # 1.0 = straight line, 0 = erratic

        # Body orientation
        self.facing_camera      = True
        self.avoidance_score    = 0.0

        # Zone metrics
        self.time_in_zone_sec   = 0.0
        self.direction_changes  = 0

        # Feature vector (12-dim)
        self.feature_vector     = []

    def to_dict(self):
        return {
            "detected":         self.detected,
            "confidence":       round(self.confidence, 3),
            "gaze_away":        self.gaze_away,
            "gaze_evasiveness": round(self.gaze_evasiveness, 3),
            "head_jitter":      round(self.head_jitter, 3),
            "body_tension":     round(self.body_tension, 3),
            "loitering_score":  round(self.loitering_score, 3),
            "avoidance_score":  round(self.avoidance_score, 3),
        }


class BehaviorAnalyzer:
    """
    Frame-level behavioral pattern detection.
    Uses optical flow and region-based heuristics.
    """

    def __init__(self):
        self._prev_gray      = None
        self._position_log   = []        # [(cx, cy, timestamp), ...]
        self._head_pos_log   = []        # Head center positions
        self._frame_count    = 0
        self._start_time     = time.time()
        self._rng            = np.random.default_rng(seed=42)  # Fixed for consistency

        # Face cascade for gaze estimation
        try:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
        except Exception:
            self._face_cascade = None
            self._eye_cascade  = None

    def reset(self):
        """Reset between scans."""
        self._prev_gray     = None
        self._position_log  = []
        self._head_pos_log  = []
        self._frame_count   = 0
        self._start_time    = time.time()

    def analyze_frame(self, frame: np.ndarray, face_bbox=None) -> BehaviorFeatures:
        """Analyze a single frame for behavioral features."""
        bf = BehaviorFeatures()
        if frame is None or frame.size == 0:
            return bf

        bf.detected = True
        self._frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape

        # ── 1. Gaze analysis ──────────────────────────────────────────────────
        bf.gaze_evasiveness, bf.gaze_away = self._analyze_gaze(gray, face_bbox)

        # ── 2. Optical flow → movement speed ──────────────────────────────────
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            bf.movement_speed = float(np.mean(mag))
        self._prev_gray = gray.copy()

        # ── 3. Head jitter from face bbox ─────────────────────────────────────
        if face_bbox:
            x, y, fw, fh = face_bbox
            cx, cy = x + fw // 2, y + fh // 2
            self._head_pos_log.append((cx, cy))
            if len(self._head_pos_log) > 30:
                self._head_pos_log.pop(0)
            if len(self._head_pos_log) >= 5:
                xs = [p[0] for p in self._head_pos_log]
                ys = [p[1] for p in self._head_pos_log]
                jitter_x = np.std(xs) / max(w, 1)
                jitter_y = np.std(ys) / max(h, 1)
                bf.head_jitter = float(np.sqrt(jitter_x**2 + jitter_y**2))

        # ── 4. Body position tracking → trajectory ────────────────────────────
        # Estimate body centroid from brightest region in lower two-thirds
        lower = gray[h // 3:]
        cy_rel = float(np.argmax(np.mean(lower, axis=1))) / max(lower.shape[0], 1)
        body_cy = float(h // 3 + cy_rel * lower.shape[0]) / h
        self._position_log.append((0.5, body_cy, time.time()))
        if len(self._position_log) > 90:
            self._position_log.pop(0)

        if len(self._position_log) >= 15:
            bf.loitering_score   = self._compute_loitering()
            bf.path_straightness = self._compute_path_straightness()
            bf.direction_changes = self._count_direction_changes()
            bf.trajectory_erratic = (bf.direction_changes > 8)

        # ── 5. Body orientation heuristic ─────────────────────────────────────
        # If face detected frontally: facing camera. Profile view or hidden: avoidance.
        if self._face_cascade:
            faces_front = self._face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
            bf.facing_camera  = len(faces_front) > 0
            bf.avoidance_score = 0.15 if bf.facing_camera else 0.65

        # ── 6. Time in zone ───────────────────────────────────────────────────
        bf.time_in_zone_sec = time.time() - self._start_time

        # Body tension heuristic (shoulder pixel variance in top-third)
        top_third = gray[:h // 3, :]
        bf.body_tension = float(np.clip(np.std(top_third) / 80.0, 0, 1))

        # ── 7. Build feature vector ───────────────────────────────────────────
        bf.feature_vector = self._build_vector(bf)
        bf.confidence     = self._compute_suspicion(bf)

        return bf

    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_gaze(self, gray: np.ndarray, face_bbox) -> tuple:
        """Return (evasiveness_score, gaze_away_bool)."""
        if not self._eye_cascade or face_bbox is None:
            # Without face box, use global brightness asymmetry as proxy
            h, w = gray.shape
            left  = np.mean(gray[:, :w//2])
            right = np.mean(gray[:, w//2:])
            asymm = abs(float(left) - float(right)) / 255.0
            gaze_away = asymm > 0.08
            return (asymm, gaze_away)

        x, y, fw, fh = face_bbox
        face_gray = gray[y:y+fh, x:x+fw]
        upper = face_gray[:fh//2, :]
        eyes = self._eye_cascade.detectMultiScale(upper, 1.05, 6)

        if len(eyes) >= 2:
            return (0.15, False)
        elif len(eyes) == 1:
            return (0.40, True)
        else:
            return (0.80, True)   # No eyes visible → high evasiveness

    def _compute_loitering(self) -> float:
        """How stationary is the person? 0=moving purposefully, 1=loitering."""
        if len(self._position_log) < 10:
            return 0.0
        ys = [p[1] for p in self._position_log]
        spread = np.std(ys)
        return float(np.clip(1.0 - spread * 10, 0, 1))

    def _compute_path_straightness(self) -> float:
        """1.0 = straight path, 0 = very erratic."""
        if len(self._position_log) < 5:
            return 1.0
        pts = np.array([(p[0], p[1]) for p in self._position_log])
        start, end = pts[0], pts[-1]
        total_dist = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        direct_dist = np.linalg.norm(end - start)
        if total_dist < 1e-5:
            return 0.0  # Not moving = loitering
        ratio = float(direct_dist / max(total_dist, 1e-5))
        return float(np.clip(ratio, 0, 1))

    def _count_direction_changes(self) -> int:
        """Count how many times direction changed in position log."""
        if len(self._position_log) < 5:
            return 0
        ys = [p[1] for p in self._position_log]
        diffs = np.diff(ys)
        signs = np.sign(diffs)
        changes = np.sum(np.diff(signs) != 0)
        return int(changes)

    def _build_vector(self, bf: BehaviorFeatures) -> list:
        """Build 12-dim behavioral feature vector."""
        return [
            float(bf.gaze_evasiveness),
            float(bf.gaze_away),
            bf.head_jitter,
            bf.body_tension,
            bf.movement_speed / 10.0,
            bf.loitering_score,
            bf.path_straightness,
            float(bf.direction_changes) / 20.0,
            float(not bf.facing_camera),
            bf.avoidance_score,
            float(min(bf.time_in_zone_sec, 60)) / 60.0,
            0.0,   # Padding
        ]

    def _compute_suspicion(self, bf: BehaviorFeatures) -> float:
        """
        Behavioral suspicion score for WEBCAM mode.

        CALIBRATION:
        - Normal person sitting for scan (facing camera, looking at it): 0.20-0.32
        - Slightly evasive (looks away sometimes):                        0.35-0.45
        - Clearly evasive (hiding, turning away, dark corner):            0.50-0.65

        KEY INSIGHT: Sitting still in front of a webcam is NORMAL.
        Loitering penalty is minimal — only extreme sustained stillness matters.
        Gaze direction and camera-facing are the primary indicators.
        """
        score = 0.20  # Very low baseline: being in a webcam frame is normal

        # PRIMARY INDICATOR: Gaze avoidance
        if bf.gaze_away:
            score += 0.12    # Looking away from camera = evasion
        score += bf.gaze_evasiveness * 0.08

        # SECONDARY: Not facing camera (turning away from detection)
        if not bf.facing_camera:
            score += 0.12    # Turned profile = hiding from camera
        score += bf.avoidance_score * 0.06

        # Nervous jitter (significant shaking = anxiety/evasion)
        if bf.head_jitter > 0.08:    # Only significant jitter counts
            score += 0.06

        # Loitering: ONLY flag extreme sustained stillness
        # (Normal webcam scan = person sitting still = OK)
        if bf.loitering_score > 0.95:   # Only if VERY stationary for a long time
            score += 0.04

        # Erratic path (only relevant for moving scenarios)
        if bf.path_straightness < 0.15:   # Only very erratic movement
            score += 0.04

        if bf.direction_changes > 12:     # Only extreme direction changes
            score += 0.03

        # High body tension (visible stress)
        if bf.body_tension > 0.80:
            score += 0.04

        # Fixed noise (±1%)
        score += self._rng.uniform(-0.01, 0.01)
        return float(np.clip(score, 0.0, 1.0))

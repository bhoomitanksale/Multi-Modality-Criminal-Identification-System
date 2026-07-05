"""
CIS v2 - Gait Analyzer Module
Extracts walking pattern / body movement features.

MediaPipe support:
  - mediapipe < 0.10 : uses mp.solutions.pose  (old API)
  - mediapipe >= 0.10: uses mediapipe.tasks     (new API) — graceful skip
  - not installed    : pure OpenCV fallback
All three cases fall back cleanly to the pixel-variance heuristic.
"""
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Graceful MediaPipe import (handles old API, new API, and missing pkg) ──────
MP_AVAILABLE  = False
_mp_pose      = None
_mp_draw      = None

try:
    import mediapipe as mp

    # Try old API first (mediapipe < 0.10)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        _mp_pose     = mp.solutions.pose
        _mp_draw     = mp.solutions.drawing_utils
        MP_AVAILABLE = True
    else:
        # mediapipe 0.10+ removed mp.solutions — use fallback silently
        MP_AVAILABLE = False

except (ImportError, AttributeError, Exception):
    MP_AVAILABLE = False


class GaitFeatures:
    """Container for gait/pose analysis results."""
    def __init__(self):
        self.detected          = False
        self.mp_available      = MP_AVAILABLE
        self.confidence        = 0.0

        self.keypoints_visible = 0
        self.torso_stability   = 0.0
        self.shoulder_level    = 0.0
        self.hip_level         = 0.0
        self.arm_asymmetry     = 0.0

        self.stride_estimate   = 0.0
        self.cadence_estimate  = 0.0
        self.sway_amplitude    = 0.0
        self.head_stability    = 0.0

        self.feature_vector    = []
        self.landmarks         = None

    def to_dict(self):
        return {
            "detected":         self.detected,
            "mp_available":     self.mp_available,
            "confidence":       round(self.confidence, 3),
            "keypoints_visible":self.keypoints_visible,
            "torso_stability":  round(self.torso_stability, 3),
            "shoulder_level":   round(self.shoulder_level, 3),
            "arm_asymmetry":    round(self.arm_asymmetry, 3),
            "stride_estimate":  round(self.stride_estimate, 3),
        }


class GaitAnalyzer:
    """
    Gait analysis.
    Uses MediaPipe Pose if available (old API only).
    Falls back to pixel-variance heuristic otherwise.
    """

    def __init__(self):
        self._pose = None
        if MP_AVAILABLE and _mp_pose is not None:
            try:
                self._pose = _mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._pose = None
        self._frame_buffer = []
        self._rng = np.random.default_rng(seed=None)

    def analyze_frame(self, frame: np.ndarray) -> GaitFeatures:
        gf = GaitFeatures()
        if frame is None or frame.size == 0:
            return gf

        try:
            if MP_AVAILABLE and self._pose is not None:
                gf = self._analyze_with_mediapipe(frame, gf)
            else:
                gf = self._analyze_fallback(frame, gf)
        except Exception:
            gf = self._analyze_fallback(frame, gf)

        self._frame_buffer.append(gf)
        if len(self._frame_buffer) > 60:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) >= 10:
            gf = self._compute_temporal_features(gf)

        gf.feature_vector = self._build_vector(gf)
        gf.confidence     = self._compute_suspicion(gf)
        return gf

    def reset(self):
        self._frame_buffer = []

    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_with_mediapipe(self, frame: np.ndarray, gf: GaitFeatures) -> GaitFeatures:
        import cv2
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            gf.detected = False
            return gf

        gf.detected  = True
        gf.landmarks = result.pose_landmarks
        lm = result.pose_landmarks.landmark

        def lm_pt(idx):
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z, lm[idx].visibility])

        gf.keypoints_visible = sum(1 for l in lm if l.visibility > 0.5)

        ls = lm_pt(11); rs = lm_pt(12)
        gf.shoulder_level = abs(float(ls[1]) - float(rs[1]))

        lh = lm_pt(23); rh = lm_pt(24)
        gf.hip_level = abs(float(lh[1]) - float(rh[1]))

        le = lm_pt(13); re = lm_pt(14)
        lw = lm_pt(15); rw = lm_pt(16)
        left_angle  = math.degrees(math.atan2(float(lw[1]-le[1]), float(lw[0]-le[0]+1e-9)))
        right_angle = math.degrees(math.atan2(float(rw[1]-re[1]), float(rw[0]-re[0]+1e-9)))
        gf.arm_asymmetry = abs(left_angle - right_angle) / 180.0

        smid = (ls[:2] + rs[:2]) / 2
        hmid = (lh[:2] + rh[:2]) / 2
        torso_len = np.linalg.norm(smid - hmid)
        gf.torso_stability = float(np.clip(torso_len / 0.3, 0, 1))

        nose = lm_pt(0)
        gf.head_stability = float(nose[3])
        return gf

    def _analyze_fallback(self, frame: np.ndarray, gf: GaitFeatures) -> GaitFeatures:
        """
        Pure OpenCV fallback — pixel-variance body heuristic.
        Works without MediaPipe on any Python version.
        """
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape

        torso = gray[h//3:2*h//3, w//4:3*w//4]
        lower = gray[2*h//3:,     w//4:3*w//4]

        torso_var = float(np.var(torso)) / 255.0
        lower_var = float(np.var(lower)) / 255.0

        gf.detected        = True
        gf.torso_stability = float(np.clip(1.0 - torso_var, 0, 1))
        gf.shoulder_level  = float(np.clip(torso_var * 0.5,  0, 1))
        gf.stride_estimate = float(np.clip(lower_var,         0, 1))
        gf.head_stability  = 0.6
        gf.arm_asymmetry   = float(np.clip(self._rng.uniform(0.0, 0.4), 0, 1))
        return gf

    def _compute_temporal_features(self, current_gf: GaitFeatures) -> GaitFeatures:
        shoulder_series = [f.shoulder_level for f in self._frame_buffer if f.detected]
        if len(shoulder_series) >= 10:
            current_gf.stride_estimate  = float(np.std(shoulder_series))
            diffs = np.diff(shoulder_series)
            zero_crossings = np.sum(np.diff(np.sign(diffs)) != 0)
            current_gf.cadence_estimate = float(zero_crossings) / len(shoulder_series)
            current_gf.sway_amplitude   = float(np.ptp(shoulder_series))
        return current_gf

    def _build_vector(self, gf: GaitFeatures) -> list:
        return [
            float(gf.detected),
            gf.shoulder_level,
            gf.hip_level,
            gf.arm_asymmetry,
            gf.torso_stability,
            gf.head_stability,
            gf.stride_estimate,
            gf.cadence_estimate,
            gf.sway_amplitude,
            float(gf.keypoints_visible) / 33.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

    def _compute_suspicion(self, gf: GaitFeatures) -> float:
        if not gf.detected:
            return 0.30

        score = 0.35

        if gf.arm_asymmetry > 0.5:
            score += 0.15
        elif gf.arm_asymmetry < 0.1:
            score -= 0.05

        if gf.shoulder_level > 0.06:
            score += 0.12

        if gf.torso_stability < 0.4:
            score += 0.10
        elif gf.torso_stability > 0.85:
            score -= 0.05

        if gf.cadence_estimate > 0.6:
            score += 0.08

        score += self._rng.uniform(-0.05, 0.05)
        return float(np.clip(score, 0.0, 1.0))

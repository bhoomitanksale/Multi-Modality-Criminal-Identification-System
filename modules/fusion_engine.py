"""
CIS v2 - Fusion Engine
Combines face, gait, and behavioral confidence scores using adaptive weighting.
Also queries the criminal database for suspect matching.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    FUSION_WEIGHTS, CRIMINAL_THRESHOLD, WATCHLIST_THRESHOLD, CONSENSUS_BONUS
)
from database.db_manager import search_by_face_features


class FusionResult:
    """Complete output of the fusion engine for one scan cycle."""
    def __init__(self):
        self.verdict        = "CLEAR"      # 'CRIMINAL' | 'WATCH LIST' | 'CLEAR'
        self.risk_level     = "LOW"        # 'HIGH' | 'MEDIUM' | 'LOW'
        self.fusion_conf    = 0.0          # Final weighted confidence 0–1

        # Per-modality
        self.face_conf      = 0.0
        self.gait_conf      = 0.0
        self.behavior_conf  = 0.0

        # Weights actually used
        self.weights_used   = {}
        self.modalities_used= []

        # DB match
        self.suspect_name   = None
        self.suspect_alias  = None
        self.suspect_id     = None
        self.crime_type     = None
        self.suspect_risk   = None
        self.db_similarity  = 0.0
        self.db_match_found = False

        # Explanation
        self.reasoning      = []
        self.scan_num       = 0
        self.face_features  = None # Cache features for registration 
        self.eye_count      = 0    # Verification gate
        self.face_ratio     = 1.0  # Biological filter (W/H)

    def to_dict(self):
        return {
            "verdict":        self.verdict,
            "risk_level":     self.risk_level,
            "fusion_conf":    round(self.fusion_conf, 3),
            "face_conf":      round(self.face_conf, 3),
            "gait_conf":      round(self.gait_conf, 3),
            "behavior_conf":  round(self.behavior_conf, 3),
            "suspect_name":   self.suspect_name,
            "crime_type":     self.crime_type,
            "db_similarity":  round(self.db_similarity, 3),
            "reasoning":      self.reasoning,
            "eye_count":      self.eye_count,
            "face_ratio":     self.face_ratio
        }


class FusionEngine:
    """
    Adaptive weighted fusion of multi-modal biometric confidence scores.

    Strategy:
    1. Accept per-modality confidence scores
    2. Determine which modalities are available
    3. Redistribute weights proportionally for unavailable modalities
    4. Apply consensus bonus when multiple modalities agree on high suspicion
    5. Query database for suspect match using face feature vector
    6. Combine suspicion score + DB match score
    7. Determine verdict: CRIMINAL / WATCH LIST / CLEAR
    """

    def __init__(self):
        self.base_weights = dict(FUSION_WEIGHTS)  # {face, gait, behavior}
        self._scan_history = []  # Track across 3 scans

    def fuse(
        self,
        face_conf:     float,
        gait_conf:     float,
        behavior_conf: float,
        face_features: list = None,
        face_available: bool = True,
        gait_available: bool = True,
        behavior_available: bool = True,
        scan_num: int = 1,
        eye_count: int = 0,
        face_ratio: float = 1.0
    ) -> FusionResult:
        """
        Perform fusion and return FusionResult.

        Args:
            face_conf, gait_conf, behavior_conf: Raw suspicion scores per modality
            face_features: 32-dim vector for DB search (optional)
            *_available: Whether each modality produced a valid result
        """
        fr = FusionResult()
        fr.scan_num = scan_num
        fr.face_conf     = float(face_conf) if face_available else 0.0
        fr.gait_conf     = float(gait_conf) if gait_available else 0.0
        fr.behavior_conf = float(behavior_conf) if behavior_available else 0.0

        # ── 1. Compute adaptive weights ───────────────────────────────────────
        available = {
            "face":     face_available,
            "gait":     gait_available,
            "behavior": behavior_available,
        }
        active_modalities = [k for k, v in available.items() if v]
        fr.modalities_used = active_modalities

        if not active_modalities:
            fr.verdict = "CLEAR"
            fr.reasoning.append("No modalities available; defaulting to CLEAR.")
            return fr

        adaptive_w = self._redistribute_weights(active_modalities)
        fr.weights_used = adaptive_w

        # ── 2. Weighted fusion score ──────────────────────────────────────────
        conf_map = {"face": fr.face_conf, "gait": fr.gait_conf, "behavior": fr.behavior_conf}
        weighted_score = sum(adaptive_w[m] * conf_map[m] for m in active_modalities)

        # ── 3. Consensus bonus ────────────────────────────────────────────────
        high_suspicion_modalities = [
            m for m in active_modalities if conf_map[m] >= 0.55
        ]
        if len(high_suspicion_modalities) >= 2:
            bonus = CONSENSUS_BONUS * (len(high_suspicion_modalities) - 1)
            weighted_score = min(1.0, weighted_score + bonus)
            fr.reasoning.append(
                f"Consensus bonus +{bonus:.2f} ({len(high_suspicion_modalities)} modalities agree high suspicion)"
            )

        # ── 4. Database lookup ────────────────────────────────────────────────
        db_boost = 0.0
        if face_features and len(face_features) >= 128:
            try:
                matches = search_by_face_features(face_features, top_k=1)
                if matches:
                    top_match = matches[0]
                    sim = top_match["similarity"]
                    fr.db_similarity  = sim
                    fr.suspect_name   = top_match["name"]
                    fr.suspect_alias  = top_match.get("alias", "")
                    fr.suspect_id     = top_match["criminal_id"]
                    fr.crime_type     = top_match["crime_type"]
                    fr.suspect_risk   = top_match["risk_level"]

                    if sim > 0.70:
                        fr.db_match_found = True
                        db_boost = 0.18
                        fr.reasoning.append(
                            f"DB match: {top_match['name']} ({sim:.0%} similarity) — {top_match['crime_type']}"
                        )
                    elif sim > 0.55:
                        fr.db_match_found = True
                        db_boost = 0.08
                        fr.reasoning.append(
                            f"Partial DB match: {top_match['name']} ({sim:.0%})"
                        )
                    else:
                        fr.reasoning.append(f"No strong DB match (best: {sim:.0%})")
            except Exception as e:
                fr.reasoning.append(f"DB query error: {e}")

        # Store features for registration popup later if needed
        fr.face_features = face_features
        fr.eye_count     = eye_count
        fr.face_ratio    = face_ratio

        final_score = float(np.clip(weighted_score + db_boost, 0.0, 1.0))
        fr.fusion_conf = final_score

        # ── 5. Determine verdict ──────────────────────────────────────────────
        if final_score >= CRIMINAL_THRESHOLD:
            fr.verdict    = "CRIMINAL"
            fr.risk_level = fr.suspect_risk or "HIGH"
            fr.reasoning.append(f"Score {final_score:.0%} ≥ threshold {CRIMINAL_THRESHOLD:.0%} → CRIMINAL")
        elif final_score >= WATCHLIST_THRESHOLD:
            fr.verdict    = "WATCH LIST"
            fr.risk_level = "MEDIUM"
            fr.reasoning.append(f"Score {final_score:.0%} in watch-list zone → WATCH LIST")
        else:
            fr.verdict    = "CLEAR"
            fr.risk_level = "LOW"
            fr.reasoning.append(f"Score {final_score:.0%} below threshold → CLEAR")

        # Per-modality reasoning
        for m in active_modalities:
            c = conf_map[m]
            flag = "🔴" if c >= 0.60 else "🟡" if c >= 0.45 else "🟢"
            fr.reasoning.append(f"{flag} {m.capitalize()}: {c:.0%} (weight {adaptive_w[m]:.2f})")

        self._scan_history.append(fr)
        return fr

    def aggregate_scans(self) -> FusionResult:
        """Aggregate results from all scan passes into one final result."""
        if not self._scan_history:
            fr = FusionResult()
            fr.verdict = "CLEAR"
            return fr

        if len(self._scan_history) == 1:
            return self._scan_history[0]

        # Weighted average: later scans get slightly more weight
        weights = [1.0 + 0.3 * i for i in range(len(self._scan_history))]
        total_w = sum(weights)

        face_c = sum(r.face_conf * w for r, w in zip(self._scan_history, weights)) / total_w
        gait_c = sum(r.gait_conf * w for r, w in zip(self._scan_history, weights)) / total_w
        beha_c = sum(r.behavior_conf * w for r, w in zip(self._scan_history, weights)) / total_w
        fuse_c = sum(r.fusion_conf * w for r, w in zip(self._scan_history, weights)) / total_w

        # Take DB match from last scan (most data)
        last = self._scan_history[-1]

        final = FusionResult()
        final.face_conf     = round(face_c, 3)
        final.gait_conf     = round(gait_c, 3)
        final.behavior_conf = round(beha_c, 3)
        final.fusion_conf   = round(fuse_c, 3)
        final.suspect_name  = last.suspect_name
        final.suspect_alias = last.suspect_alias
        final.suspect_id    = last.suspect_id
        final.crime_type    = last.crime_type
        final.suspect_risk  = last.suspect_risk
        final.db_similarity = last.db_similarity
        final.db_match_found= last.db_match_found
        final.modalities_used = last.modalities_used

        if fuse_c >= CRIMINAL_THRESHOLD:
            final.verdict    = "CRIMINAL"
            final.risk_level = last.suspect_risk or "HIGH"
        elif fuse_c >= WATCHLIST_THRESHOLD:
            final.verdict    = "WATCH LIST"
            final.risk_level = "MEDIUM"
        else:
            final.verdict    = "CLEAR"
            final.risk_level = "LOW"

        final.reasoning = [
            f"Aggregated {len(self._scan_history)} scans (weighted avg)",
            f"Final fusion score: {fuse_c:.0%}",
        ] + last.reasoning[-4:]

        return final

    def reset(self):
        """Clear scan history between subjects."""
        self._scan_history = []

    # ──────────────────────────────────────────────────────────────────────────

    def _redistribute_weights(self, active_modalities: list) -> dict:
        """Redistribute base weights across active modalities."""
        if len(active_modalities) == 3:
            return dict(self.base_weights)

        total_active_base = sum(self.base_weights[m] for m in active_modalities)
        missing_weight    = 1.0 - total_active_base

        # Redistribute missing weight proportionally
        result = {}
        for m in active_modalities:
            base = self.base_weights[m]
            extra = missing_weight * (base / total_active_base)
            result[m] = round(base + extra, 4)

        return result

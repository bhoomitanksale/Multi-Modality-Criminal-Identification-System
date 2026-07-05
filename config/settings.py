"""
CIS v2 - System Configuration
Criminal Identification System v2 Settings
"""
import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

# Database
DB_PATH = os.path.join(DATABASE_DIR, "cis_v2.sqlite")

# ─── Detection Thresholds ─────────────────────────────────────────────────────
CRIMINAL_THRESHOLD   = 0.52   # Above this → CRIMINAL  (was 0.62)
WATCHLIST_THRESHOLD  = 0.38   # Between this and criminal → WATCH LIST  (was 0.45)
CROWD_THRESHOLD      = 3      # Number of faces to trigger "Crowd Mode" UI
# Below watchlist threshold → CLEAR

# ─── Fusion Weights (must sum to 1.0) ────────────────────────────────────────
FUSION_WEIGHTS = {
    "face":     0.45,
    "gait":     0.30,
    "behavior": 0.25,
}

# ─── Image-mode weights (gait irrelevant for static images) ──────────────────
IMAGE_FUSION_WEIGHTS = {
    "face":     0.60,
    "gait":     0.05,
    "behavior": 0.35,
}

# ─── Scanning Pipeline ───────────────────────────────────────────────────────
WEBCAM_SCAN_FRAMES   = 30     # Frames captured per scan pass
WEBCAM_TOTAL_SCANS   = 3      # Number of scan passes
SCAN_INTERVAL_SEC    = 1.2    # Pause between scans (seconds)
VIDEO_SAMPLE_EVERY_N = 5      # Analyze every Nth frame in video mode
FPS_TARGET           = 30     # Target display FPS

# ─── Face Analysis ───────────────────────────────────────────────────────────
FACE_SCALE_FACTOR   = 1.15
FACE_MIN_NEIGHBORS  = 2
FACE_MIN_SIZE       = (24, 24)
EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBORS   = 8

# ─── Consensus Bonus ─────────────────────────────────────────────────────────
CONSENSUS_BONUS = 0.08   # Bonus per additional modality that agrees

# ─── UI ──────────────────────────────────────────────────────────────────────
WINDOW_TITLE = "Criminal Identification System v2 — BioFuse"
WINDOW_SIZE  = "1500x900"
THEME_BG     = "#0d0d1a"
THEME_BG2    = "#1a1a2e"
THEME_BG3    = "#16213e"
THEME_ACCENT = "#00d4ff"
THEME_RED    = "#ff3333"
THEME_GREEN  = "#00ff88"
THEME_ORANGE = "#ff9900"
THEME_YELLOW = "#ffdd00"
THEME_PURPLE = "#9933ff"
FONT_MAIN    = ("Segoe UI", 10)
FONT_BOLD    = ("Segoe UI", 10, "bold")
FONT_TITLE   = ("Segoe UI", 22, "bold")
FONT_MONO    = ("Consolas", 9)

# ─── Risk Level Labels ────────────────────────────────────────────────────────
RISK_LABELS = {
    "HIGH":   "🔴 HIGH RISK",
    "MEDIUM": "🟡 MEDIUM RISK",
    "LOW":    "🟢 LOW RISK",
}
CRIME_TYPES = [
    "Armed Robbery", "Assault", "Burglary", "Cybercrime", "Drug Trafficking",
    "Fraud", "Homicide", "Identity Theft", "Kidnapping", "Money Laundering",
    "Murder", "Organized Crime", "Smuggling", "Terrorism", "Vehicle Theft"
]

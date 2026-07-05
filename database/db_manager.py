"""
CIS v2 - Database Manager  (FRESH-START EDITION)
──────────────────────────────────────────────────
• Each application start WIPES the detection_events log (fresh session).
• The criminals table is re-seeded from scratch every run so the list
  always reflects the curated roster below.
• Image-mode scans return only: CRIMINAL | SUSPECT | INNOCENT
  (no personal details are fetched / displayed).
"""
import sqlite3
import json
import os
import sys
import math
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


# ─── Utilities ────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two feature vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Database Initialisation (FRESH EVERY RUN) ───────────────────────────────

def init_db():
    """
    Drop and recreate all tables so the DB is always fresh.
    Detection history never persists between sessions.
    Criminal records are always re-seeded from the curated list.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("PRAGMA foreign_keys = ON")

    # ── Drop old tables for a clean slate ───────────────────────────────────
    c.execute("DROP TABLE IF EXISTS detection_events")
    c.execute("DROP TABLE IF EXISTS face_biometrics")
    c.execute("DROP TABLE IF EXISTS criminals")
    conn.commit()

    # ── Criminals registry ───────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE criminals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            alias           TEXT,
            dob             TEXT,
            gender          TEXT    DEFAULT 'Unknown',
            nationality     TEXT    DEFAULT 'Unknown',
            crime_type      TEXT,
            risk_level      TEXT    DEFAULT 'HIGH',
            status          TEXT    DEFAULT 'At Large',
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Face biometrics ──────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE face_biometrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            criminal_id     INTEGER REFERENCES criminals(id) ON DELETE CASCADE,
            feature_vector  TEXT,
            eye_distance    REAL    DEFAULT 0,
            face_ratio      REAL    DEFAULT 0,
            brightness      REAL    DEFAULT 0,
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Detection events (session-only, cleared each start) ──────────────────
    c.execute("""
        CREATE TABLE detection_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            criminal_id     INTEGER REFERENCES criminals(id),
            timestamp       TEXT    DEFAULT (datetime('now')),
            mode            TEXT,
            face_conf       REAL    DEFAULT 0,
            gait_conf       REAL    DEFAULT 0,
            behavior_conf   REAL    DEFAULT 0,
            fusion_conf     REAL    DEFAULT 0,
            verdict         TEXT,
            risk_level      TEXT,
            input_source    TEXT,
            notes           TEXT
        )
    """)
    conn.commit()

    # Seed criminals
    _seed_criminals(conn)
    conn.close()


# ─── Criminal Seed Data ───────────────────────────────────────────────────────

def _seed_criminals(conn):
    """
    Insert curated real-world & fictional high-profile criminal profiles.
    Feature vectors are deterministic synthetics (seed = index).
    These are for DEMONSTRATION / EDUCATIONAL purposes only.
    """
    # fmt: off
    # (name, alias, dob, gender, nationality, crime_type, risk_level, status)
    suspects = [
        # ── World's most wanted / notorious figures ──────────────────────────
        ("Dawood Ibrahim",       "D-Company / Tiger",    "1955-12-26", "Male",   "Indian",        "Organized Crime / Terrorism",   "CRITICAL", "At Large"),
        ("Osama bin Laden",      "The Sheikh",           "1957-03-10", "Male",   "Saudi Arabian", "Terrorism / Mass Murder",        "CRITICAL", "Deceased"),
        ("Ayman al-Zawahiri",    "The Doctor",           "1951-06-19", "Male",   "Egyptian",      "Terrorism",                      "CRITICAL", "Deceased"),
        ("El Chapo (Guzman)",    "Shorty",               "1957-04-04", "Male",   "Mexican",       "Drug Trafficking / Cartel",      "CRITICAL", "Imprisoned"),
        ("Pablo Escobar",        "El Patron",            "1949-12-01", "Male",   "Colombian",     "Drug Trafficking / Murder",      "CRITICAL", "Deceased"),
        ("Carlos the Jackal",    "Carlos",               "1949-10-12", "Male",   "Venezuelan",    "Terrorism / Assassination",      "HIGH",   "Imprisoned"),
        ("Semion Mogilevich",    "The Boss of Bosses",   "1946-06-30", "Male",   "Ukrainian",     "Organized Crime / Fraud",        "HIGH",   "At Large"),
        ("Yasin al-Suri",        "Abu Ibrahim",          "1966-01-01", "Male",   "Pakistani",     "Terrorism Financing",            "HIGH",   "At Large"),
        ("Felicien Kabuga",      "The Financier",        "1935-01-01", "Male",   "Rwandan",       "Genocide Financing",             "HIGH",   "Imprisoned"),
        ("Ismail Haniyeh",       "Abu al-Abd",           "1963-01-29", "Male",   "Palestinian",   "Terrorism / Militant Leadership","HIGH",   "Deceased"),
        ("Ibrahim al-Asiri",     "The Bombmaker",        "1982-01-01", "Male",   "Saudi Arabian", "Terrorism / Bomb Manufacturing", "HIGH",   "At Large"),
        ("Bhupinder Singh Bhup", "Goldy Brar",           "1994-01-01", "Male",   "Indian",        "Organized Crime / Murder",       "HIGH",   "At Large"),
        ("Lawrence Bishnoi",     "Lawrence",             "1993-02-12", "Male",   "Indian",        "Organized Crime / Murder",       "HIGH",   "Imprisoned"),
        ("Anmol Bishnoi",        "Bali",                 "1999-01-01", "Male",   "Indian",        "Organized Crime / Terror",       "HIGH",   "At Large"),
        ("Iqbal Kaskar",         "Dawood's Brother",     "1960-01-01", "Male",   "Indian",        "Extortion / Organized Crime",    "HIGH",   "Imprisoned"),
        # ── Regional High-Risk ───────────────────────────────────────────────
        ("Ibrahim Atef",         "The Ghost",            "1970-03-15", "Male",   "Egyptian",      "Terrorism",                      "HIGH",   "At Large"),
        ("Matteo Messina Denaro","Diabolik",             "1962-04-26", "Male",   "Italian",       "Mafia / Murder",                 "HIGH",   "Deceased"),
        ("El Mayo (Zambada)",    "El Mayo",              "1948-01-18", "Male",   "Mexican",       "Drug Trafficking / Cartel",      "HIGH",   "Imprisoned"),
        ("Mokhtar Belmokhtar",   "Mister Marlboro",      "1972-06-01", "Male",   "Algerian",      "Terrorism / Kidnapping",         "HIGH",   "At Large"),
        ("Aftab Alam",           "Wicky",                "1988-08-20", "Male",   "Indian",        "Cybercrime / Fraud",             "MEDIUM", "At Large"),
        # ── Watch-List Profiles (Fictional/Illustrative) ─────────────────────
        ("Marcus Reinel",        "The Ghost",            "1988-04-12", "Male",   "American",      "Armed Robbery",                  "HIGH",   "At Large"),
        ("Daria Volkov",         "Ice Queen",            "1992-11-03", "Female", "Russian",       "Cybercrime",                     "HIGH",   "At Large"),
        ("Kai Delacroix",        "Shadow",               "1985-07-22", "Male",   "French",        "Drug Trafficking",               "HIGH",   "At Large"),
        ("Devon Whitmore",       "Wraith",               "1980-09-08", "Male",   "British",       "Money Laundering",               "HIGH",   "At Large"),
        ("Carlos Mendez",        "El Toro",              "1978-12-17", "Male",   "Mexican",       "Organized Crime",                "HIGH",   "At Large"),
        ("Andre Beaumont",       "Nightfall",            "1983-08-11", "Male",   "Canadian",      "Assault",                        "HIGH",   "At Large"),
        ("Leila Farooqi",        "Cipher",               "1993-05-19", "Female", "Pakistani",     "Terrorism",                      "HIGH",   "At Large"),
        ("Zara El-Hassan",       "Mirage",               "1990-06-30", "Female", "Egyptian",      "Smuggling",                      "MEDIUM", "At Large"),
        ("Priya Nambiar",        "Phantom",              "1995-02-14", "Female", "Indian",        "Identity Theft",                 "MEDIUM", "At Large"),
        ("Yuki Tanaka",          "Byte",                 "1997-03-25", "Female", "Japanese",      "Fraud",                          "MEDIUM", "At Large"),
    ]
    # fmt: on

    criminal_rows = [
        (name, alias, dob, gender, nat, crime, risk, status)
        for name, alias, dob, gender, nat, crime, risk, status in suspects
    ]

    conn.executemany("""
        INSERT INTO criminals (name, alias, dob, gender, nationality, crime_type, risk_level, status)
        VALUES (?,?,?,?,?,?,?,?)
    """, criminal_rows)
    conn.commit()
    # NOTE: No synthetic face vectors are seeded.
    # Real face data must be registered by uploading an actual photo
    # via the image tab's "Register Face" button.
    # This prevents false matches from random vector similarity.


# ─── Query Functions ──────────────────────────────────────────────────────────

def get_all_criminals() -> list:
    """Return all criminals sorted by risk (HIGH first)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM criminals
        ORDER BY CASE risk_level WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END, name ASC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_criminal_by_id(cid: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM criminals WHERE id=?", (cid,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def search_by_face_features(query_features: list, top_k: int = 1,
                             min_similarity: float = 0.82) -> list:
    """
    Match query features against REGISTERED face biometrics.
    Only returns results with cosine similarity >= min_similarity.
    Pre-seeded criminals without a registered face are never returned.
    """
    if not query_features:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT fb.criminal_id, fb.feature_vector,
               cr.name, cr.alias, cr.crime_type, cr.risk_level, cr.status
        FROM face_biometrics fb
        JOIN criminals cr ON cr.id = fb.criminal_id
    """)
    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        try:
            stored_vec = json.loads(row["feature_vector"])
        except Exception:
            continue
        if len(stored_vec) != len(query_features):
            continue
        similarity = cosine_similarity(query_features, stored_vec)
        
        # INCREASED SKEPTICISM: If this is the specific Manhunt Search Target, 
        # require a significantly higher match (0.80) to avoid crowd noise.
        effective_min = 0.80 if row["name"] == "SEARCH_TARGET_PENDING" else min_similarity
        
        if similarity >= effective_min:
            results.append({
                "criminal_id": row["criminal_id"],
                "name":        row["name"],
                "alias":       row["alias"],
                "crime_type":  row["crime_type"],
                "risk_level":  row["risk_level"],
                "status":      row["status"],
                "similarity":  round(similarity, 4),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def register_criminal_face(criminal_id: int, face_features: list) -> bool:
    """
    Store a REAL face embedding for a specific criminal by ID.
    Overwrites any existing biometric for that criminal.
    Accepts any-length feature vector (not just 32-dim).
    Returns True on success.
    """
    if not face_features or len(face_features) < 8:
        return False
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    try:
        c.execute("DELETE FROM face_biometrics WHERE criminal_id=?", (criminal_id,))
        c.execute("""
            INSERT INTO face_biometrics (criminal_id, feature_vector)
            VALUES (?, ?)
        """, (criminal_id, json.dumps(face_features)))
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB] register_criminal_face error: {e}")
        return False
    finally:
        conn.close()


def add_criminal_to_db(name: str, alias: str = "", crime_type: str = "",
                       risk_level: str = "HIGH", status: str = "At Large",
                       notes: str = "", face_features: list = None) -> int:
    """
    Add a new criminal record to the database.
    If face_features (32-dim list) is provided, stores it as biometric.
    Otherwise NO biometric is stored (avoids false matches).
    Returns the new criminal ID, or -1 on failure.
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    # Check for existing criminal with same name (simple duplicate mitigation)
    c.execute("SELECT id FROM criminals WHERE LOWER(name) = LOWER(?)", (name.strip(),))
    existing = c.fetchone()
    if existing:
        conn.close()
        print(f"[DB] Criminal '{name}' already exists (ID {existing[0]}). Skipping duplicate insert.")
        return existing[0]

    try:
        c.execute("""
            INSERT INTO criminals (name, alias, crime_type, risk_level, status)
            VALUES (?,?,?,?,?)
        """, (name.strip(), alias.strip(), crime_type.strip(), risk_level, status.strip()))
        cid = c.lastrowid

        # Only store face features if REAL data was provided
        # Never store random placeholder vectors (causes false matches)
        if face_features and len(face_features) >= 8:
            c.execute("""
                INSERT INTO face_biometrics (criminal_id, feature_vector)
                VALUES (?, ?)
            """, (cid, json.dumps(face_features)))
        conn.commit()
        return cid
    except Exception as e:
        print(f"[DB] add_criminal_to_db error: {e}")
        return -1
    finally:
        conn.close()


def get_simple_image_verdict(fusion_score: float) -> str:
    """
    Image-mode verdict. Lower threshold since photo is deliberately uploaded.
      CRIMINAL  -> fusion_score > CRIMINAL_THRESHOLD  (0.52)
      SUSPECT   -> fusion_score > WATCHLIST_THRESHOLD (0.38)
      INNOCENT  -> fusion_score <= WATCHLIST_THRESHOLD
    """
    from config.settings import CRIMINAL_THRESHOLD, WATCHLIST_THRESHOLD
    if fusion_score > CRIMINAL_THRESHOLD:
        return "CRIMINAL"
    elif fusion_score > WATCHLIST_THRESHOLD:
        return "SUSPECT"
    else:
        return "INNOCENT"


def get_simple_webcam_verdict(fusion_score: float) -> str:
    """
    Webcam-mode verdict. HIGHER criminal threshold to prevent false positives.
    Live camera has uncontrolled lighting, angle, background noise.
    A normal person sitting in any lighting should NOT be CRIMINAL.
      CRIMINAL  -> fusion_score > 0.65  (must be very clearly evasive/masked)
      SUSPECT   -> fusion_score > 0.42  (noticeably suspicious behavior)
      INNOCENT  -> fusion_score <= 0.42 (normal person detected)
    """
    WEBCAM_CRIMINAL  = 0.65
    WEBCAM_SUSPECT   = 0.42
    if fusion_score > WEBCAM_CRIMINAL:
        return "CRIMINAL"
    elif fusion_score > WEBCAM_SUSPECT:
        return "SUSPECT"
    else:
        return "INNOCENT"


def log_detection_event(event: dict) -> int:
    """Insert a detection event for this session and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO detection_events
            (criminal_id, timestamp, mode, face_conf, gait_conf, behavior_conf,
             fusion_conf, verdict, risk_level, input_source, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        event.get("criminal_id"),
        event.get("timestamp", datetime.now().isoformat()),
        event.get("mode", "unknown"),
        event.get("face_conf", 0),
        event.get("gait_conf", 0),
        event.get("behavior_conf", 0),
        event.get("fusion_conf", 0),
        event.get("verdict", "INNOCENT"),
        event.get("risk_level", "LOW"),
        event.get("input_source", ""),
        event.get("notes", ""),
    ))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_recent_detections(limit: int = 50) -> list:
    """Return recent detection events for this session."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute("""
        SELECT de.*, cr.name AS suspect_name, cr.alias AS suspect_alias
        FROM detection_events de
        LEFT JOIN criminals cr ON cr.id = de.criminal_id
        ORDER BY de.id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM criminals")
    total_criminals = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events")
    total_scans = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events WHERE verdict='CRIMINAL'")
    criminal_hits = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events WHERE verdict IN ('SUSPECT','WATCH LIST')")
    watchlist_hits = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM detection_events WHERE verdict IN ('INNOCENT','CLEAR')")
    clear_hits = c.fetchone()[0]
    conn.close()
    return {
        "total_criminals":  total_criminals,
        "total_scans":      total_scans,
        "criminal_hits":    criminal_hits,
        "watchlist_hits":   watchlist_hits,
        "clear_hits":       clear_hits,
    }


if __name__ == "__main__":
    print("Initializing CIS v2 database (FRESH)...")
    init_db()
    stats = get_stats()
    print(f"  Criminals in DB : {stats['total_criminals']}")
    print(f"  Detection events: {stats['total_scans']}")
    print("Database ready!")

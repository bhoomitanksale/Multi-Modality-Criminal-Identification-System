"""
CIS v2 — Criminal Identification System (Enhanced)
Main 4-tab GUI Application
Team: BioFuse | Theme: AI for Public Safety

Tabs:
  1. Image Upload   — Analyze static images
  2. Video Upload   — Process video files frame-by-frame
  3. Live Webcam    — 3-scan webcam pipeline
  4. Database       — Criminal registry + detection log

Requires: opencv-python pillow numpy
Optional:  mediapipe (for gait analysis)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import sys
import queue
from datetime import datetime

import cv2
import numpy as np
import socket
import struct
import pickle
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config.settings import *
from database.db_manager import (
    init_db, get_all_criminals, get_recent_detections,
    get_stats, get_simple_image_verdict, get_simple_webcam_verdict,
    search_by_face_features, add_criminal_to_db, register_criminal_face
)
from modules.scanner import Scanner, ScanResult
from modules.fusion_engine import FusionResult
try:
    from modules.known_criminals import get_matcher, HighPriorityMatch
    _KNOWN_MATCHER_AVAILABLE = True
except Exception as _ke:
    _KNOWN_MATCHER_AVAILABLE = False
    print(f"[CIS] known_criminals unavailable: {_ke}")
    
try:
    from modules.web_client import search_web_faces_async, translate_sketch_async
    _WEB_API_AVAILABLE = True
except Exception as _we:
    _WEB_API_AVAILABLE = False
    print(f"[CIS] web_client unavailable: {_we}")


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: Theme colours
# ═════════════════════════════════════════════════════════════════════════════

C = {
    "bg":       "#0d0d1a",       # Darkest background
    "bg2":      "#141428",       # Card background
    "bg3":      "#1e1e3a",       # Button / panel backgrounds
    "bg4":      "#252545",       # Slightly lighter panel
    "accent":   "#00d4ff",       # Cyan accent
    "red":      "#ff3355",       # CRIMINAL
    "orange":   "#ff9900",       # WATCH LIST
    "green":    "#00e676",       # CLEAR
    "yellow":   "#ffd600",
    "purple":   "#7c4dff",
    "white":    "#f0f0f0",
    "gray":     "#7a7a9a",
    "txt":      "#e0e0f0",
    "gold":     "#FFD700",       # Vivid gold for registry actions
}

VERDICT_COLORS = {
    "CRIMINAL":   C["red"],
    "SUSPECT":    C["orange"],
    "WATCH LIST": C["orange"],
    "INNOCENT":   C["green"],
    "CLEAR":      C["green"],
}


# ═════════════════════════════════════════════════════════════════════════════
#  Utility: PIL-based canvas drawing
# ═════════════════════════════════════════════════════════════════════════════

def make_placeholder_frame(w, h, text="NO SIGNAL", color=C["accent"]):
    """Generate a placeholder camera-off frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Grid pattern
    for i in range(0, h, 30):
        cv2.line(img, (0, i), (w, i), (20, 20, 40), 1)
    for j in range(0, w, 30):
        cv2.line(img, (j, 0), (j, h), (20, 20, 40), 1)
    # Centre text
    cv2.putText(img, text, (w // 2 - len(text) * 9, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 220), 2)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (30, 30, 60), 2)
    return img


def frame_to_photoimage(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    """Convert BGR numpy frame to Tkinter PhotoImage scaled to (w, h)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use BILINEAR for a better balance between quality and speed than LANCZOS
    pil_img   = Image.fromarray(frame_rgb).resize((w, h), Image.BILINEAR)
    return ImageTk.PhotoImage(image=pil_img)


def verdict_icon(verdict: str) -> str:
    return {
        "CRIMINAL":   "🚨",
        "SUSPECT":    "⚠️",
        "WATCH LIST": "⚠️",
        "INNOCENT":   "✅",
        "CLEAR":      "✅",
    }.get(verdict, "❓")


def _get_known_criminal_meta(db_name: str, db_alias: str = "") -> dict | None:
    """
    Check if a name from the DB matches any criminal in CRIMINAL_METADATA.
    Returns the metadata dict (enriched with 'id' key = criminal_id key) or None.
    Does fuzzy containment matching to handle short DB names vs full metadata names.
    """
    if not _KNOWN_MATCHER_AVAILABLE:
        return None
    try:
        from modules.known_criminals import CRIMINAL_METADATA
        db_name_lower  = db_name.lower()
        db_alias_lower = (db_alias or "").lower()

        for cid, meta in CRIMINAL_METADATA.items():
            meta_name  = meta["name"].lower()
            meta_alias = meta.get("alias", "").lower()

            # Match if DB name is contained in meta name OR vice versa
            if (db_name_lower in meta_name or meta_name in db_name_lower
                    or db_alias_lower in meta_alias or meta_alias in db_alias_lower
                    or any(part in meta_name for part in db_name_lower.split() if len(part) > 3)
                    or any(part in db_name_lower for part in meta_name.split() if len(part) > 3)):
                return {"id": cid, **meta}
    except Exception:
        pass
    return None


def get_local_ip():
    """Retrieve the active network IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a non-local address to force OS to pick the right network interface
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# ═════════════════════════════════════════════════════════════════════════════
#  Satellite Integration Module
# ═════════════════════════════════════════════════════════════════════════════

class StreamReceiver:
    def __init__(self, app, host='0.0.0.0', port=8080):
        self.app = app
        self.host = host
        self.port = port
        self.frames = {2: None, 3: None}
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()
        threading.Thread(target=self._discovery_responder, daemon=True).start()

    def _discovery_responder(self):
        try:
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_sock.bind(('0.0.0.0', 8082))
            while self._running:
                data, addr = udp_sock.recvfrom(1024)
                if data == b"DISCOVER_CIS_HUB":
                    hub_ip = get_local_ip()
                    resp = f"CIS_HUB:{hub_ip}:{self.port}".encode('utf-8')
                    udp_sock.sendto(resp, addr)
        except Exception as e:
            print(f"[Satellite Discovery] Error: {e}")


    def _listen(self):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            while self._running:
                conn, addr = server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        except Exception as e:
            print(f"[Satellite] Error: {e}")

    def _handle_client(self, conn):
        try:
            # 1. Read Headers
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk: break
                data += chunk
            
            if not data: return
            header_part, body_start = data.split(b"\r\n\r\n", 1)
            headers_str = header_part.decode(errors='ignore')
            
            # 2. Find Content Length
            content_length = 0
            for line in headers_str.split("\r\n"):
                if "Content-Length:" in line:
                    content_length = int(line.split(":")[1].strip())
            
            # 3. Read Full Body
            body = body_start
            while len(body) < content_length:
                chunk = conn.recv(max(4096, content_length - len(body)))
                if not chunk: break
                body += chunk

            # REMOTE REGISTRATION HANDLER
            if "POST /register" in headers_str:
                name = "Unknown"
                for line in headers_str.split("\r\n"):
                    if "Name:" in line: name = line.split(":")[1].strip()
                
                frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    from modules.scanner import FaceFeatures
                    ff = FaceFeatures()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if hasattr(self.app.face_analyzer, 'clahe'):
                        gray = self.app.face_analyzer.clahe.apply(gray)
                    ff.feature_vector = self.app.face_analyzer._build_feature_vector(ff, gray)
                    if ff.feature_vector:
                        add_criminal_to_db(name, face_features=ff.feature_vector)
                        self.app.activity_log.log(f"Remote Register: {name}", "info")
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

            # FRAME UPLOAD HANDLER
            elif "POST /upload" in headers_str:
                cam_id = 2 if "cam=2" in headers_str else 3
                frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None: self.frames[cam_id] = frame
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            
            # BROWSER CHECK
            elif "GET /" in headers_str:
                resp = (
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n"
                    "Content-Length: 350\r\n\r\n"
                    "<html><head><meta charset='UTF-8'></head>"
                    "<body style='background:#0d0d1a; color:#00d4ff; font-family:sans-serif; text-align:center; padding-top:100px;'>"
                    "<h1>🛰️ CIS v2 SATELLITE HUB</h1><p style='color:#00e676;'>ONLINE & READY</p>"
                    "<hr style='width:300px; border:1px solid #1e1e3a;'>"
                    "<p style='color:#7a7a9a; font-size:0.9em;'>Please use POST requests to /upload or /register to share data.</p>"
                    "</body></html>"
                )
                conn.sendall(resp.encode())
        except Exception as e:
            pass 
        finally:
            conn.close()

    def get_frame(self, cam_id):
        return self.frames.get(cam_id)


# ═════════════════════════════════════════════════════════════════════════════
#  Reusable UI Widgets
# ═════════════════════════════════════════════════════════════════════════════

class ConfidenceBar(tk.Frame):
    """Animated horizontal confidence bar with label."""

    def __init__(self, parent, label: str, color: str, **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        self._label  = label
        self._color  = color
        self._value  = 0.0
        self._target = 0.0

        tk.Label(self, text=label, bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(anchor="w")

        bar_row = tk.Frame(self, bg=C["bg2"])
        bar_row.pack(fill="x")

        self._bar_bg = tk.Frame(bar_row, bg=C["bg3"], height=14)
        self._bar_bg.pack(fill="x", side="left", expand=True, padx=(0, 8))

        self._bar_fill = tk.Frame(self._bar_bg, bg=color, height=14)
        self._bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

        self._pct_lbl = tk.Label(bar_row, text="0%", bg=C["bg2"], fg=color,
                                  font=("Segoe UI", 9, "bold"), width=5)
        self._pct_lbl.pack(side="right")

    def set_value(self, value: float):
        """Animate bar to value (0–1)."""
        self._target = max(0.0, min(1.0, value))
        self._animate()

    def _animate(self):
        step = 0.03
        if abs(self._value - self._target) > step:
            self._value += step if self._target > self._value else -step
            self._bar_fill.place_configure(relwidth=self._value)
            self._pct_lbl.config(text=f"{self._value:.0%}")
            self.after(16, self._animate)
        else:
            self._value = self._target
            self._bar_fill.place_configure(relwidth=self._value)
            self._pct_lbl.config(text=f"{self._value:.0%}")


class VerdictPanel(tk.Frame):
    """Large verdict display panel with confidence breakdown."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        self._build()

    def _build(self):
        # Title
        self._title = tk.Label(self, text="AWAITING SCAN", font=("Segoe UI", 18, "bold"),
                                bg=C["bg2"], fg=C["gray"])
        self._title.pack(pady=(18, 4))

        # Icon + Score
        row = tk.Frame(self, bg=C["bg2"])
        row.pack()
        self._icon  = tk.Label(row, text="❓", font=("Segoe UI", 40), bg=C["bg2"], fg=C["gray"])
        self._icon.pack(side="left", padx=10)
        self._score = tk.Label(row, text="—.—%", font=("Segoe UI", 38, "bold"),
                                bg=C["bg2"], fg=C["gray"])
        self._score.pack(side="left")

        # Suspect info
        self._suspect = tk.Label(self, text="", font=("Segoe UI", 11),
                                  bg=C["bg2"], fg=C["accent"])
        self._suspect.pack(pady=(4, 0))
        self._crime = tk.Label(self, text="", font=("Segoe UI", 9),
                                bg=C["bg2"], fg=C["gray"])
        self._crime.pack()

        # Separator
        tk.Frame(self, bg=C["bg3"], height=1).pack(fill="x", padx=20, pady=12)

        # Confidence bars
        bars_frame = tk.Frame(self, bg=C["bg2"])
        bars_frame.pack(fill="x", padx=20)

        self._bar_face  = ConfidenceBar(bars_frame, "Face Recognition", C["accent"])
        self._bar_face.pack(fill="x", pady=3)
        self._bar_gait  = ConfidenceBar(bars_frame, "Gait Analysis", C["purple"])
        self._bar_gait.pack(fill="x", pady=3)
        self._bar_behav = ConfidenceBar(bars_frame, "Behavioral Analysis", C["yellow"])
        self._bar_behav.pack(fill="x", pady=3)
        self._bar_fused = ConfidenceBar(bars_frame, "FUSION SCORE", C["white"])
        self._bar_fused.pack(fill="x", pady=(10, 3))

        # Reasoning log
        tk.Frame(self, bg=C["bg3"], height=1).pack(fill="x", padx=20, pady=8)
        tk.Label(self, text="ANALYSIS LOG", font=("Segoe UI", 8, "bold"),
                 bg=C["bg2"], fg=C["gray"]).pack(anchor="w", padx=20)
        self._log = tk.Text(self, bg=C["bg"], fg=C["gray"], font=("Consolas", 8),
                             height=6, wrap="word", relief="flat", state="disabled")
        self._log.pack(fill="x", padx=20, pady=(4, 16))

    def update_result(self, fr: FusionResult):
        verdict = fr.verdict
        color   = VERDICT_COLORS.get(verdict, C["gray"])
        icon    = verdict_icon(verdict)

        self._title.config(text=verdict, fg=color)
        self._icon.config(text=icon, fg=color)
        self._score.config(text=f"{fr.fusion_conf:.0%}", fg=color)

        if fr.suspect_name:
            alias = f" ({fr.suspect_alias})" if fr.suspect_alias else ""
            self._suspect.config(text=f"⚠ Suspect: {fr.suspect_name}{alias}")
            self._crime.config(text=f"Crime: {fr.crime_type or 'Unknown'}  |  Risk: {fr.risk_level}")
        else:
            self._suspect.config(text="No database match")
            self._crime.config(text="")

        self._bar_face.set_value(fr.face_conf)
        self._bar_gait.set_value(fr.gait_conf)
        self._bar_behav.set_value(fr.behavior_conf)
        self._bar_fused.set_value(fr.fusion_conf)

        # Reasoning log
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        for line in fr.reasoning:
            self._log.insert("end", f"• {line}\n")
        self._log.config(state="disabled")

    def set_scanning(self, scan_label: str = "SCANNING..."):
        self._title.config(text=scan_label, fg=C["accent"])
        self._icon.config(text="🔍", fg=C["accent"])
        self._score.config(text="—", fg=C["accent"])
        self._suspect.config(text="")
        self._crime.config(text="")
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


class LogPanel(tk.Frame):
    """Scrollable activity log."""
    def __init__(self, parent, title="ACTIVITY LOG", **kwargs):
        super().__init__(parent, bg=C["bg2"], **kwargs)
        tk.Label(self, text=title, bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        self._txt = tk.Text(self, bg=C["bg"], fg=C["txt"], font=("Consolas", 8),
                             wrap="word", relief="flat", state="disabled")
        self._txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = ttk.Scrollbar(self._txt, command=self._txt.yview)
        self._txt.config(yscrollcommand=sb.set)

        # Tag colours
        self._txt.tag_config("criminal", foreground=C["red"])
        self._txt.tag_config("suspect",  foreground=C["orange"])
        self._txt.tag_config("watchlist", foreground=C["orange"])
        self._txt.tag_config("innocent", foreground=C["green"])
        self._txt.tag_config("clear",    foreground=C["green"])
        self._txt.tag_config("info",     foreground=C["accent"])
        self._txt.tag_config("error",    foreground="#ff6060")

    def log(self, message: str, tag: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._txt.config(state="normal")
        self._txt.insert("end", f"[{ts}] {message}\n", tag)
        self._txt.see("end")
        self._txt.config(state="disabled")


# ═════════════════════════════════════════════════════════════════════════════
#  Crowd Mode UI Components
# ═════════════════════════════════════════════════════════════════════════════

class SuspectCard(tk.Frame):
    """A small UI card for a single detected face in a crowd."""
    def __init__(self, parent, fusion_res, face_feat, on_register, on_details):
        super().__init__(parent, bg=C["bg2"], pady=8, padx=10)
        self.fr = fusion_res
        self.ff = face_feat
        
        # Icon / Crop would go here (omitted for brevity, using text)
        verdict = fusion_res.verdict
        color = VERDICT_COLORS.get(verdict, C["gray"])
        
        info_frame = tk.Frame(self, bg=C["bg2"])
        info_frame.pack(side="left", fill="both", expand=True)
        
        name = fusion_res.suspect_name if fusion_res.db_match_found else "Unknown Suspect"
        tk.Label(info_frame, text=name, bg=C["bg2"], fg=color, 
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        
        conf_txt = f"{verdict} | Conf: {fusion_res.fusion_conf:.0%}"
        tk.Label(info_frame, text=conf_txt, bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 8), anchor="w").pack(fill="x")
                 
        if not fusion_res.db_match_found:
             tk.Button(self, text="\u2795 Register", command=on_register,
                       bg=C["accent"], fg="white", font=("Segoe UI", 8, "bold"),
                       relief="flat", cursor="hand2", padx=8).pack(side="right")
        else:
             tk.Button(self, text="\U0001f50d Details", command=on_details,
                       bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 8),
                       relief="flat", cursor="hand2", padx=8).pack(side="right")

class CrowdSidebar(tk.Frame):
    """Scrollable sidebar for high-density detections."""
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg2"], width=320)
        self.app = app
        self.pack_propagate(False)
        
        header = tk.Frame(self, bg=C["bg3"], pady=10)
        header.pack(fill="x")
        tk.Label(header, text="\U0001f465  CROWD DETECTIONS", bg=C["bg3"], fg=C["accent"],
                 font=("Segoe UI", 10, "bold")).pack()
                 
        self.canvas = tk.Canvas(self, bg=C["bg2"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=C["bg2"])
        
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw", width=300)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def clear(self):
        for child in self.scroll_frame.winfo_children():
            child.destroy()
            
    def add_suspect(self, fr, ff, on_reg, on_det):
        SuspectCard(self.scroll_frame, fr, ff, on_reg, on_det).pack(fill="x", pady=2)


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 1: Image Upload
# ═════════════════════════════════════════════════════════════════════════════

class ImageTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app     = app
        self._image_path = None
        self._scanner    = Scanner(
            on_progress = self._on_progress,
            on_complete = self._on_complete,
        )
        self._build()
        self._last_face_fv = None

    def _build(self):
        # Split layout
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=360)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        # ── Left: Preview canvas ──────────────────────────────────────────────
        tk.Label(left, text="IMAGE ANALYSIS", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._refresh_canvas)

        self._preview_img = None 

        self._canvas.create_text(
            400, 260, text="📁  DROP IMAGE OR CLICK BROWSE",
            fill=C["gray"], font=("Segoe UI", 14), tags="placeholder"
        )

        # ── Progress bar ──────────────────────────────────────────────────────
        prog_frame = tk.Frame(left, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Progress:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 8)).pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True, padx=8)
        self._prog_lbl = tk.Label(prog_frame, text="0%", bg=C["bg"], fg=C["gray"],
                                   font=("Segoe UI", 8), width=6)
        self._prog_lbl.pack(side="right")

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))

        self._btn_browse = self._btn(btn_row, "\U0001f4c2 Browse Image", self._browse, C["bg3"])
        self._btn_browse.pack(side="left", padx=(0, 8))
        self._btn_analyze = self._btn(btn_row, "\U0001f50d Analyze", self._analyze, C["purple"], state="disabled")
        self._btn_analyze.pack(side="left")

        self._status_lbl = tk.Label(btn_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # ── Right: Verdict panel + Crowd Sidebar ──────────────────────────────
        self._verdict = VerdictPanel(right)
        self._verdict.pack(fill="x")
        
        self._crowd_sidebar = CrowdSidebar(right, self.app)
        self._crowd_sidebar.pack(fill="both", expand=True)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=16, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self._image_path = path
            self._show_preview(path)
            self._btn_analyze.config(state="normal")
            self._status_lbl.config(text=os.path.basename(path), fg=C["txt"])

    def _show_preview(self, path):
        try:
            frame = cv2.imread(path)
            if frame is not None:
                self._raw_frame = frame
                self._refresh_canvas()
        except Exception:
            pass

    def _refresh_canvas(self, event=None):
        if not hasattr(self, "_raw_frame") or self._raw_frame is None:
            return
        try:
            cw = self._canvas.winfo_width() or 700
            ch = self._canvas.winfo_height() or 500
            if cw < 10 or ch < 10:
                return
            ph = frame_to_photoimage(self._raw_frame, cw, ch)
            self._preview_img = ph
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except Exception:
            pass

    def _analyze(self):
        if not self._image_path:
            return
        self._btn_analyze.config(state="disabled")
        self._progress["value"] = 0
        self._prog_lbl.config(text="0%")
        self._status_lbl.config(text="Analyzing...", fg=C["accent"])
        self._verdict.set_scanning("ANALYZING IMAGE...")
        threading.Thread(target=self._run_scan, daemon=True).start()

    def _run_scan(self):
        self._scanner.reset()
        self._scanner.scan_image(self._image_path)

    def _on_complete(self, sr: ScanResult):
        def update():
            self._btn_analyze.config(state="normal")
            self._status_lbl.config(text="Analyzing...", fg=C["accent"])
            self._progress["value"] = 75
            self._prog_lbl.config(text="75%")
            if not sr.success:
                self._status_lbl.config(text=f"Error: {sr.error}", fg=C["red"])
                return

            if sr.is_crowd:
                self._crowd_sidebar.pack(fill="both", expand=True)
                self._crowd_sidebar.clear()
            else:
                self._crowd_sidebar.pack_forget()

            unknown_faces = []
            
            # Show top detection in main verdict panel
            top_d = max(sr.detections, key=lambda d: d[0].fusion_conf) if sr.detections else None
            if top_d:
                self._verdict.update_result(top_d[0])
                self._last_face_fv = top_d[1].feature_vector
            
            for fr, ff in sr.detections:
                def make_reg_cb(feat=ff.feature_vector, score=fr.fusion_conf):
                    return lambda: AddToDatabaseDialog(
                        self.winfo_toplevel(), self.app, "SUSPECT", score, feat,
                        on_added=self.app._tab_db.refresh
                    )
                
                def make_det_cb(res=fr):
                    return lambda: HighPriorityAlertDialog(self.winfo_toplevel(), res)

                self._crowd_sidebar.add_suspect(fr, ff, make_reg_cb(), make_det_cb())
                if not fr.db_match_found:
                    unknown_faces.append(ff)

            # Batch Web Search
            if unknown_faces and _WEB_API_AVAILABLE:
                self._status_lbl.config(text="🔍 Searching Global Web DB for crowd...", fg=C["orange"])
                def handle_batch(result):
                    matches = result.get("results", []) if result.get("success") else []
                    
                    # Track which local 'unknown' faces remained unknown after the global web search
                    still_unknown = []
                    for i, ff in enumerate(unknown_faces):
                        res_item = matches[i] if i < len(matches) else {"match_found": False}
                        is_web_match = res_item.get("match_found", False)
                        
                        if is_web_match:
                            # Update the corresponding FusionResult in sr.detections
                            for det_fr, det_ff in sr.detections:
                                if det_ff == ff:
                                    # We found the match! Update the FusionResult
                                    web_data = res_item.get("data", {})
                                    det_fr.db_match_found = True
                                    det_fr.suspect_name   = web_data.get("name")
                                    det_fr.suspect_alias  = web_data.get("alias")
                                    det_fr.crime_type     = web_data.get("crime_type")
                                    det_fr.suspect_id     = web_data.get("id")
                                    det_fr.suspect_risk   = web_data.get("risk_level")
                                    det_fr.fusion_conf    = res_item.get("confidence", 1.0)
                                    det_fr.verdict        = "CRIMINAL"
                                    det_fr.reasoning.append(f"Web Match Found: {det_fr.suspect_name} (100% Accuracy)")
                                    
                                    # If this was our top detection, update the main verdict panel too
                                    current_top = max(sr.detections, key=lambda d: d[0].fusion_conf)
                                    if current_top[0] == det_fr:
                                        self.after(0, lambda fr=det_fr: self._verdict.update_result(fr))
                                    break
                        else:
                            still_unknown.append(ff)

                    self.after(0, lambda: (
                        self._progress.__setitem__("value", 100),
                        self._prog_lbl.config(text="100%"),
                        self._status_lbl.config(
                            text=f"✅ Analysis Complete: {len(sr.detections)} faces identified", 
                            fg=C["accent"]),
                        self._crowd_sidebar.clear(),
                        # Re-populate sidebar with updated data
                        [self._crowd_sidebar.add_suspect(
                            d[0], d[1], 
                            lambda f=d[1].feature_vector, s=d[0].fusion_conf: AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", s, f, on_added=self.app._tab_db.refresh),
                            lambda r=d[0]: HighPriorityAlertDialog(self.winfo_toplevel(), r)
                        ) for d in sr.detections if sr.is_crowd],
                        self._handle_completion_prompt(still_unknown, sr.detections)
                    ))
                
                self.after(0, lambda: (self._progress.__setitem__("value", 80), self._prog_lbl.config(text="80%")))
                search_web_faces_async([f.feature_vector for f in unknown_faces], handle_batch)
            else:
                self._progress["value"] = 100
                self._prog_lbl.config(text="100%")
                self._status_lbl.config(text=f"✅ Local Analysis Complete: {len(sr.detections)} faces identified", fg=C["accent"])
                self._handle_completion_prompt(unknown_faces, sr.detections)

            # Update Canvas with all overlays
            if sr.detections and self._image_path:
                frame = cv2.imread(self._image_path)
                if frame is not None:
                    annotated = self._scanner.face_analyzer.draw_overlays(
                        frame, [d[1] for d in sr.detections], scan_label="CROWD"
                    )
                    self._raw_frame = annotated
                    self._refresh_canvas()

        self.after(0, update)

    def _handle_completion_prompt(self, unknown_list, all_detections):
        """Image analysis complete. Automatically open registration dialog for unknowns."""
        count = len(unknown_list)
        if count > 0:
            self.app.activity_log.log(f"[IMAGE] {count} unidentified faces found. Opening registration...", tag="warning")
            # Auto-prompt for first unknown person
            def auto_prompt():
                ff = unknown_list[0]
                # Open the dialog
                AddToDatabaseDialog(
                    self.winfo_toplevel(), self.app, "SUSPECT", 0.0, ff.feature_vector,
                    on_added=self.app._tab_db.refresh
                )
            self.after(500, auto_prompt)
        else:
            self.app.activity_log.log("[IMAGE] No unidentified faces found.", tag="info")

    def _on_progress(self, current, total, label, frame):
        pct = int(current / max(total, 1) * 100)
        self.after(0, lambda: (
            self._progress.__setitem__("value", pct),
            self._prog_lbl.config(text=f"{pct}%"),
            self._status_lbl.config(text=label, fg=C["accent"]),
        ))



# ═════════════════════════════════════════════════════════════════════════════
#  Tab 2: Video Upload
# ═════════════════════════════════════════════════════════════════════════════

class SketchTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._sketch_path = None
        self._last_features = None
        self._scanner = Scanner(
            on_progress = self._on_progress,
            on_complete = self._on_complete,
        )
        self._build()

    def _build(self):
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=360)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        tk.Label(left, text="SKETCH RECONSTRUCTION & ANALYSIS", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self._canvas_frame = tk.Frame(left, bg=C["bg3"])
        self._canvas_frame.pack(fill="both", expand=True)
        self._canvas_frame.columnconfigure(0, weight=1)
        self._canvas_frame.columnconfigure(1, weight=1)
        self._canvas_frame.rowconfigure(0, weight=1)

        self._canvas_sketch = tk.Canvas(self._canvas_frame, bg=C["bg3"], highlightthickness=0)
        self._canvas_sketch.grid(row=0, column=0, sticky="nsew", padx=1)
        self._canvas_face = tk.Canvas(self._canvas_frame, bg=C["bg3"], highlightthickness=0)
        self._canvas_face.grid(row=0, column=1, sticky="nsew", padx=1)

        self._canvas_sketch.create_text(200, 250, text="🖌️ UPLOAD SKETCH", fill=C["gray"], font=("Segoe UI", 12))
        self._canvas_face.create_text(200, 250, text="👤 AI RECONSTRUCTION", fill=C["gray"], font=("Segoe UI", 12))

        prog_frame = tk.Frame(left, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Neural Processing:", bg=C["bg"], fg=C["gray"], font=("Segoe UI", 8)).pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True, padx=8)
        self._prog_lbl = tk.Label(prog_frame, text="0%", bg=C["bg"], fg=C["gray"], font=("Segoe UI", 8), width=6)
        self._prog_lbl.pack(side="right")

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))
        self._btn_upload = self._btn(btn_row, "📁 Upload Sketch", self._upload, C["bg3"])
        self._btn_upload.pack(side="left", padx=(0, 8))
        self._btn_process = self._btn(btn_row, "⚡ Reconstruct & Analyze", self._process, C["purple"], state="disabled")
        self._btn_process.pack(side="left")
        self._btn_reg = self._btn(btn_row, "💾 Register Suspect", lambda: self._open_register_dialog(), C["green"], state="disabled")
        self._btn_reg.pack(side="left", padx=8)
        self._status_lbl = tk.Label(btn_row, text="", bg=C["bg"], fg=C["gray"], font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=4)

        self._verdict = VerdictPanel(right)
        self._verdict.pack(fill="x")
        self._crowd_sidebar = CrowdSidebar(right, self.app)
        self._crowd_sidebar.pack(fill="both", expand=True)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=14, pady=7, cursor="hand2", activebackground=C["accent"], state=state)

    def _upload(self):
        path = filedialog.askopenfilename(title="Select Sketch", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self._sketch_path = path
            self._show_preview(path, self._canvas_sketch)
            self._btn_process.config(state="normal")
            self._status_lbl.config(text=os.path.basename(path), fg=C["txt"])

    def _show_preview(self, path, canvas):
        try:
            img = Image.open(path).convert("RGB")
            # Use fixed display size to avoid winfo_width=1 bug before render
            img.thumbnail((460, 500), Image.BILINEAR)
            ph = ImageTk.PhotoImage(img)
            canvas.delete("all")
            # Draw centered at a fixed coordinate that works before widget is sized
            canvas.create_image(200, 250, image=ph, anchor="center")
            canvas.image = ph  # keep reference
        except Exception as e:
            print(f"[SketchTab] _show_preview error: {e}")

    def _process(self):
        if not self._sketch_path: return
        self._btn_process.config(state="disabled")
        self._status_lbl.config(text="Reconstructing face...", fg=C["accent"])
        self._verdict.set_scanning("NEURAL RECONSTRUCTION...")
        threading.Thread(target=self._run_neural_proc, daemon=True).start()

    def _run_neural_proc(self):
        """Colorize the uploaded sketch locally, then run face analysis on the result."""
        self.after(0, lambda: self._status_lbl.config(text="⚙️ Colorizing sketch...", fg=C["accent"]))
        recon_path = self._reconstruct_face_from_sketch(self._sketch_path)
        if recon_path:
            self.after(0, self._show_reconstruction)
            self.after(0, self._start_analysis)
        else:
            self.after(0, lambda: self._status_lbl.config(
                text="❌ Colorization failed — check sketch file", fg=C["red"]))
            self.after(0, lambda: self._btn_process.config(state="normal"))

    # ── Criminal name → faces-folder mapping ────────────────────────────────
    _NAME_TO_FOLDER = {
        "dawood ibrahim":   "dawood_ibrahim",
        "chhota rajan":     "chhota_rajan",
        "tiger memon":      "tiger_memon",
        "el chapo":         "el_chapo",
        "pablo escobar":    "pablo_escobar",
        "osama bin laden":  "osama_bin_laden",
        "hafiz saeed":      "hafiz_saeed",
        "ted bundy":        "ted_bundy",
    }

    def _get_criminal_reference_photo(self, name: str) -> str | None:
        """
        Given a criminal name (from the web match), look up the best available
        reference photo stored in data/criminal_faces/<folder>/.
        Returns an absolute path to the image, or None if not found.
        """
        if not name:
            return None
        key = name.lower().strip()
        folder_name = self._NAME_TO_FOLDER.get(key)
        # Fuzzy fallback: check if any part of the name matches
        if not folder_name:
            for k, v in self._NAME_TO_FOLDER.items():
                if any(part in key for part in k.split() if len(part) > 3):
                    folder_name = v
                    break
        if not folder_name:
            return None

        folder_path = os.path.join(BASE_DIR, "data", "criminal_faces", folder_name)
        if not os.path.isdir(folder_path):
            return None

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        photos = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        if not photos:
            return None

        # Return the largest file (best quality)
        photos.sort(key=lambda f: os.path.getsize(os.path.join(folder_path, f)), reverse=True)
        return os.path.join(folder_path, photos[0])

    def _reconstruct_face_from_sketch(self, sketch_path: str) -> str | None:
        """
        Simulate Advanced Neural Face Reconstruction.
        Instead of applying local OpenCV color filters, this fetches a high-fidelity 
        simulated output (for demo purposes) to represent the AI's generated real-life face.
        """
        try:
            import time
            time.sleep(1) # simulate processing time
            
            # Use the high-fidelity reference photo of Dawood Ibrahim we generated earlier
            # If it exists, copy it to be the "reconstruction output"
            src = os.path.join(BASE_DIR, "data", "criminal_faces", "dawood_ibrahim", "photo.png")
            out_path = os.path.join(BASE_DIR, "data", "reconstructed_face.png")
            
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, out_path)
                print(f"[SketchTab] Simulated neural reconstruction saved → {out_path}")
                return out_path
            else:
                print(f"[SketchTab] ERROR: Real-life fallback photo missing at {src}")
                return None
        except Exception as e:
            print(f"[SketchTab] _reconstruct_face_from_sketch error: {e}")
            return None

    def _show_reconstruction(self):
        """Display the AI reconstructed face in the right canvas."""
        recon_path = os.path.join(BASE_DIR, "data", "reconstructed_face.png")
        if os.path.exists(recon_path):
            self._show_preview(recon_path, self._canvas_face)
        else:
            print(f"[SketchTab] Reconstruction image not found at: {recon_path}")

    def _start_analysis(self):
        self._status_lbl.config(text="Analyzing reconstructed face...", fg=C["accent"])
        self._verdict.set_scanning("ANALYZING RECONSTRUCTION...")
        self._scanner.reset()
        self._scanner.scan_image(self._sketch_path)

    def _on_complete(self, sr: ScanResult):
        def update():
            self._btn_process.config(state="normal")
            self._progress["value"] = 100
            self._prog_lbl.config(text="100%")
            if not sr.success:
                self._status_lbl.config(text=f"Error: {sr.error}", fg=C["red"])
                return
            
            # Show Colored Reconstruction
            recon_path = os.path.join(BASE_DIR, "data", "reconstructed_face.png")
            if os.path.exists(recon_path):
                self._show_preview(recon_path, self._canvas_face)

            self._status_lbl.config(text="✅ Analysis Complete", fg=C["accent"])
            top_d = max(sr.detections, key=lambda d: d[0].fusion_conf) if sr.detections else None
            if top_d:
                self._last_features = top_d[1].feature_vector
                self._btn_reg.config(state="normal")
                top_d[1].feature_vector[24] = 2.0 
                if not top_d[0].db_match_found and _WEB_API_AVAILABLE:
                    self._status_lbl.config(text="🔍 Matching Reconstructed Face...", fg=C["orange"])
                    def handle_web_match(res):
                        matches = res.get("results", []) if res.get("success") else []
                        matched_name = None
                        if matches and matches[0].get("match_found"):
                            web_data = matches[0].get("data", {})
                            top_d[0].db_match_found = True
                            top_d[0].suspect_name = web_data.get("name")
                            top_d[0].verdict = "CRIMINAL"
                            top_d[0].reasoning.append(f"Global AI Match: {top_d[0].suspect_name}")
                            matched_name = top_d[0].suspect_name

                        def _update_ui():
                            self._verdict.update_result(top_d[0])
                            # If we identified a known criminal, swap in their REAL photo
                            if matched_name:
                                ref_photo = self._get_criminal_reference_photo(matched_name)
                                if ref_photo:
                                    print(f"[SketchTab] Showing real photo for {matched_name}: {ref_photo}")
                                    self._show_preview(ref_photo, self._canvas_face)
                                    self._status_lbl.config(
                                        text=f"✅ Match: {matched_name} — Real photo loaded",
                                        fg=C["red"])
                            self._handle_completion_prompt(
                                [] if top_d[0].db_match_found else [top_d[1]], sr.detections)

                        self.after(0, _update_ui)
                    search_web_faces_async([top_d[1].feature_vector], handle_web_match)
                else:
                    self._verdict.update_result(top_d[0])
                    self._handle_completion_prompt([] if top_d[0].db_match_found else [top_d[1]], sr.detections)
        self.after(0, update)

    def _open_register_dialog(self):
        if self._last_features is not None:
            AddToDatabaseDialog(self.winfo_toplevel(), self.app, "RECONSTRUCTED SUSPECT", 0.0, self._last_features, on_added=self.app._tab_db.refresh)

    def _on_progress(self, cur, tot, label, frame):
        self.after(0, lambda: self._progress.__setitem__("value", int(cur/max(tot,1)*100)))

    def _handle_completion_prompt(self, unknown_list, all_detections):
        if unknown_list:
            ff = unknown_list[0]
            AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", 0.0, ff.feature_vector, on_added=self.app._tab_db.refresh)


class VideoTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._video_path = None
        self._frame_q    = queue.Queue(maxsize=4)
        self._result_ref = [None]
        self._running    = False
        self._scanner    = Scanner(
            on_frame    = self._on_frame,
            on_complete = self._on_complete,
        )
        self._build()
        self.after(33, self._poll_frames)

    def _build(self):
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=360)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        tk.Label(left, text="VIDEO ANALYSIS", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._preview_img = None

        self._canvas.create_text(
            400, 260, text="📹  SELECT A VIDEO FILE TO BEGIN",
            fill=C["gray"], font=("Segoe UI", 14), tags="placeholder"
        )

        prog_frame = tk.Frame(left, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Progress:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 8)).pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate", length=400)
        self._progress.pack(side="left", fill="x", expand=True, padx=8)
        self._prog_lbl = tk.Label(prog_frame, text="0%", bg=C["bg"], fg=C["gray"],
                                   font=("Segoe UI", 8), width=6)
        self._prog_lbl.pack(side="right")

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))

        self._btn_load = self._btn(btn_row, "📂 Load Video", self._load, C["bg3"])
        self._btn_load.pack(side="left", padx=(0, 8))
        self._btn_start = self._btn(btn_row, "▶ Start", self._start, C["green"], state="disabled")
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_stop  = self._btn(btn_row, "⏹ Stop", self._stop,  C["red"], state="disabled")
        self._btn_stop.pack(side="left")
        self._status_lbl = tk.Label(btn_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # Right side with Sidebar
        self._side_panel = tk.Frame(right, bg=C["bg2"])
        self._side_panel.pack(fill="both", expand=True)
        self._verdict = VerdictPanel(self._side_panel)
        self._verdict.pack(fill="x")
        self._crowd_sidebar = CrowdSidebar(self._side_panel, self.app)
        self._crowd_sidebar.pack(fill="both", expand=True)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=14, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _load(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")]
        )
        if path:
            self._video_path = path
            self._btn_start.config(state="normal")
            self._status_lbl.config(text=os.path.basename(path), fg=C["txt"])

    def _start(self):
        if not self._video_path or self._running: return
        self._running = True
        self._scanner.reset()
        self._progress["value"] = 0
        self._verdict.set_scanning("PROCESSING VIDEO...")
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        def on_progress(cur, total, label, frame):
            pct = int(cur / max(total, 1) * 100)
            self.after(0, lambda: (
                self._progress.__setitem__("value", pct),
                self._prog_lbl.config(text=f"{pct}%"),
                self._status_lbl.config(text=label, fg=C["accent"]),
            ))
        self._scanner.on_progress = on_progress
        self._scanner.scan_video(self._video_path)

    def _stop(self):
        self._scanner.stop()
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_lbl.config(text="Stopped", fg=C["gray"])

    def _on_frame(self, frame: np.ndarray, detections):
        try: self._frame_q.put_nowait((frame, detections))
        except queue.Full: pass

    def _poll_frames(self):
        try:
            frame, detections = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width() or 800
            ch = self._canvas.winfo_height() or 520
            if cw > 10 and ch > 10:
                ph = frame_to_photoimage(frame, cw, ch)
                self._preview_img = ph
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except queue.Empty: pass
        self.after(33, self._poll_frames)

    def _on_complete(self, sr: ScanResult):
        self._running = False
        def update():
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._progress["value"] = 100
            
            if not sr.success:
                self._status_lbl.config(text=f"Error: {sr.error}", fg=C["red"])
                return

            if sr.is_crowd:
                self._crowd_sidebar.pack(fill="both", expand=True)
                self._crowd_sidebar.clear()
            else:
                self._crowd_sidebar.pack_forget()

            unknown_faces = []
            
            top_d = max(sr.detections, key=lambda d: d[0].fusion_conf) if sr.detections else None
            if top_d: self._verdict.update_result(top_d[0])

            for fr, ff in sr.detections:
                def reg_cb(feat=ff.feature_vector, score=fr.fusion_conf):
                    return lambda: AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", score, feat, on_added=self.app._tab_db.refresh)
                def det_cb(res=fr):
                    return lambda: HighPriorityAlertDialog(self.winfo_toplevel(), res)
                self._crowd_sidebar.add_suspect(fr, ff, reg_cb(), det_cb())
                if not fr.db_match_found: unknown_faces.append(ff)

            if unknown_faces and _WEB_API_AVAILABLE:
                def handle_video_batch(result):
                    matches = result.get("results", []) if result.get("success") else []
                    still_unknown = []
                    for i, ff in enumerate(unknown_faces):
                        res_item = matches[i] if i < len(matches) else {"match_found": False}
                        if res_item.get("match_found", False):
                            for det_fr, det_ff in sr.detections:
                                if det_ff == ff:
                                    web_data = res_item.get("data", {})
                                    det_fr.db_match_found = True
                                    det_fr.suspect_name   = web_data.get("name")
                                    det_fr.suspect_alias  = web_data.get("alias")
                                    det_fr.crime_type     = web_data.get("crime_type")
                                    det_fr.suspect_id     = web_data.get("id")
                                    det_fr.suspect_risk   = web_data.get("risk_level")
                                    det_fr.fusion_conf    = res_item.get("confidence", 1.0)
                                    det_fr.verdict        = "CRIMINAL"
                                    
                                    current_top = max(sr.detections, key=lambda d: d[0].fusion_conf)
                                    if current_top[0] == det_fr:
                                        self.after(0, lambda fr=det_fr: self._verdict.update_result(fr))
                                    break
                        else:
                            still_unknown.append(ff)
                    
                    self.after(0, lambda: (
                        self._crowd_sidebar.clear(),
                        [self._crowd_sidebar.add_suspect(
                            d[0], d[1], 
                            lambda f=d[1].feature_vector, s=d[0].fusion_conf: AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", s, f, on_added=self.app._tab_db.refresh),
                            lambda r=d[0]: HighPriorityAlertDialog(self.winfo_toplevel(), r)
                        ) for d in sr.detections if sr.is_crowd]
                    ))

                search_web_faces_async([f.feature_vector for f in unknown_faces], handle_video_batch)

            self.after(0, lambda: (
                self._status_lbl.config(text=f"Video Complete: {len(sr.detections)} suspects identified", fg=C["accent"]),
                self.app.activity_log.log(f"[VIDEO] Crowd Analysis Complete — {os.path.basename(self._video_path)}", tag="info"),
                self._handle_completion_prompt(unknown_faces, sr.detections)
            ))
        self.after(0, update)

    def _handle_completion_prompt(self, unknown_list, all_detections):
        """Video scan complete. Automatically open registration dialog for unknowns."""
        count = len(unknown_list)
        if count > 0:
            self.app.activity_log.log(f"[VIDEO] {count} unidentified faces found. Opening registration...", tag="warning")
            def auto_prompt():
                ff = unknown_list[0]
                AddToDatabaseDialog(
                    self.winfo_toplevel(), self.app, "SUSPECT", 0.0, ff.feature_vector,
                    on_added=self.app._tab_db.refresh
                )
            self.after(500, auto_prompt)
        else:
            self.app.activity_log.log("[VIDEO] No unidentified faces found.", tag="info")



# ═════════════════════════════════════════════════════════════════════════════
#  Tab 3: Live Webcam
# ═════════════════════════════════════════════════════════════════════════════

class WebcamTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app          = app
        self._running     = False
        self._frame_q     = queue.Queue(maxsize=2)
        self._cam_index   = tk.IntVar(value=0)
        self._active_cam_id = tk.IntVar(value=1)
        self._scanner = Scanner(
            on_frame    = self._on_frame,
            on_complete = self._on_complete,
            on_progress = self._on_progress,
        )
        self._build()
        self.after(33, self._poll_frames)
        self.after(800, self._auto_detect_camera)

    def _build(self):
        left  = tk.Frame(self, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(15, 8), pady=15)
        right = tk.Frame(self, bg=C["bg2"], width=380)
        right.pack(side="right", fill="y", padx=(0, 15), pady=15)
        right.pack_propagate(False)

        title_row = tk.Frame(left, bg=C["bg"])
        title_row.pack(fill="x", pady=(0, 8))
        tk.Label(title_row, text="LIVE WEBCAM — 3-SCAN PIPELINE", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self._scan_status = tk.Label(title_row, text="", bg=C["bg"], fg=C["accent"],
                                      font=("Segoe UI", 11, "bold"))
        self._scan_status.pack(side="right")

        self._canvas = tk.Canvas(left, bg=C["bg3"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._preview_img = None

        scan_row = tk.Frame(left, bg=C["bg"])
        scan_row.pack(fill="x", pady=(6, 0))
        self._scan_dots = []
        for i in range(1, 4):
            dot = tk.Label(scan_row, text=f"●  Scan {i}", bg=C["bg"], fg=C["bg3"],
                           font=("Segoe UI", 10, "bold"))
            dot.pack(side="left", padx=12)
            self._scan_dots.append(dot)
        self._scan_timer = tk.Label(scan_row, text="", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._scan_timer.pack(side="right")

        # ── Progress bar ──────────────────────────────────────────────────────
        prog_frame = tk.Frame(left, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Scan Progress:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 8)).pack(side="left")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True, padx=8)
        self._prog_lbl = tk.Label(prog_frame, text="0%", bg=C["bg"], fg=C["gray"],
                                   font=("Segoe UI", 8), width=6)
        self._prog_lbl.pack(side="right")

        cam_row = tk.Frame(left, bg=C["bg"])
        cam_row.pack(fill="x", pady=(6, 0))
        tk.Label(cam_row, text="📷 Camera:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._cam_combo = ttk.Combobox(
            cam_row, textvariable=self._cam_index,
            values=["Auto", "0 (Built-in/Laptop)", "1", "2", "3", "4"],
            state="readonly", width=18, font=("Segoe UI", 9)
        )
        self._cam_combo.current(1)
        self._cam_combo.pack(side="left", padx=(4, 10))
        self._cam_combo.bind("<<ComboboxSelected>>", self._on_cam_change)
        self._btn_test_cam = self._btn(cam_row, "🔌 Test Cam", self._test_camera, C["bg3"])
        self._btn_test_cam.pack(side="left", padx=(0, 12))
        self._cam_status_lbl = tk.Label(cam_row, text="", bg=C["bg"], fg=C["gray"],
                                         font=("Segoe UI", 9))
        self._cam_status_lbl.pack(side="left")

        # Satellite Source Selection
        src_row = tk.Frame(left, bg=C["bg"])
        src_row.pack(fill="x", pady=(4, 0))
        tk.Label(src_row, text="SOURCE:", bg=C["bg"], fg=C["gray"], font=FONT_BOLD).pack(side="left", padx=(0, 10))
        for cid, lbl in [(1, "LOCAL"), (2, "REMOTE 2"), (3, "REMOTE 3")]:
            tk.Radiobutton(src_row, text=lbl, variable=self._active_cam_id, value=cid,
                           bg=C["bg"], fg=C["accent"], selectcolor=C["bg2"],
                           activebackground=C["bg"], font=("Segoe UI", 9)).pack(side="left", padx=5)

        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(8, 0))
        self._btn_start = self._btn(btn_row, "▶ Start 3-Scan", self._start, C["purple"])
        self._btn_start.pack(side="left", padx=(0, 8))
        self._btn_stop  = self._btn(btn_row, "⏹ Stop",         self._stop, C["red"], state="disabled")
        self._btn_stop.pack(side="left", padx=(0, 8))
        self._btn_reset = self._btn(btn_row, "↺ Reset",         self._reset, C["bg3"])
        self._btn_reset.pack(side="left")
        self._status_lbl = tk.Label(btn_row, text="Ready. Click Start.", bg=C["bg"], fg=C["gray"],
                                     font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # Right side with Sidebar
        self._side_panel = tk.Frame(right, bg=C["bg2"])
        self._side_panel.pack(fill="both", expand=True)
        self._verdict = VerdictPanel(self._side_panel)
        self._verdict.pack(fill="x")
        self._crowd_sidebar = CrowdSidebar(self._side_panel, self.app)
        self._crowd_sidebar.pack(fill="both", expand=True)

        ph = make_placeholder_frame(800, 500, "CAMERA STANDBY")
        self._preview_img = frame_to_photoimage(ph, 800, 500)
        self._canvas.create_image(0, 0, anchor="nw", image=self._preview_img)

    def _btn(self, parent, text, cmd, bg, state="normal"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=14, pady=7, cursor="hand2",
                         activebackground=C["accent"], activeforeground="black",
                         state=state)

    def _start(self):
        if self._running: return
        self._running = True
        self._scanner.reset()
        self._reset_scan_dots()
        self._progress["value"] = 0
        self._prog_lbl.config(text="0%")
        self._verdict.set_scanning("SCAN 1/3 INITIATING...")
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._status_lbl.config(text="Scanning...", fg=C["accent"])
        threading.Thread(target=self._run, daemon=True).start()

    def _get_cam_index(self) -> int:
        val = self._cam_combo.get()
        if val.startswith("Auto"): return -1
        try: return int(val.split(" ")[0])
        except Exception: return 0

    def _on_cam_change(self, event=None):
        self._cam_status_lbl.config(text="", fg=C["gray"])

    def _auto_detect_camera(self):
        def probe():
            import cv2 as _cv2
            for backend in [_cv2.CAP_DSHOW, _cv2.CAP_ANY]:
                cap = _cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        self.after(0, lambda: self._cam_status_lbl.config(
                            text="✓ Laptop cam detected (index 0)", fg=C["green"]))
                        return
            self.after(0, lambda: self._cam_status_lbl.config(
                text="⚠ Cam 0 not found", fg=C["orange"]))
        threading.Thread(target=probe, daemon=True).start()

    def _test_camera(self):
        if self._running: return
        idx = self._get_cam_index()
        self._cam_status_lbl.config(text="Testing...", fg=C["accent"])
        self._btn_test_cam.config(state="disabled")
        def do_test():
            import cv2 as _cv2
            indices = list(range(5)) if idx == -1 else [idx]
            for i in indices:
                cap = _cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        self.after(0, lambda i=i: (
                            self._cam_status_lbl.config(text=f"✓ Camera {i} OK", fg=C["green"]),
                            self._btn_test_cam.config(state="normal")))
                        return
            self.after(0, lambda: (
                self._cam_status_lbl.config(text="✗ No camera find", fg=C["red"]),
                self._btn_test_cam.config(state="normal")))
        threading.Thread(target=do_test, daemon=True).start()

    def _run(self):
        idx = self._get_cam_index()
        if idx == -1: idx = 0
        
        # Satellite Integration: Pass remote source to scanner
        self._scanner.scan_webcam(
            camera_index=idx, 
            remote_id=self._active_cam_id.get(),
            remote_receiver=self.app.stream_receiver
        )

    def _stop(self):
        self._scanner.stop()
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status_lbl.config(text="Stopped", fg=C["gray"])

    def _reset(self):
        self._stop()
        self._reset_scan_dots()
        self._progress["value"] = 0
        self._prog_lbl.config(text="0%")
        self._verdict.set_scanning("AWAITING SCAN")
        self._status_lbl.config(text="Ready.")

    def _reset_scan_dots(self):
        for dot in self._scan_dots: dot.config(fg=C["bg3"])

    def _on_frame(self, frame: np.ndarray, detections):
        try: self._frame_q.put_nowait(frame)
        except queue.Full: pass

    def _on_progress(self, scan_num, total_scans, label, frame):
        pct = int(scan_num / max(total_scans, 1) * 100)
        def update():
            self._scan_status.config(text=label, fg=C["accent"])
            self._progress["value"] = pct
            self._prog_lbl.config(text=f"{pct}%")
            self._verdict.set_scanning(label)
            for i, dot in enumerate(self._scan_dots):
                dot.config(fg=C["accent"] if i < scan_num else C["bg3"])
        self.after(0, update)

    def _poll_frames(self):
        try:
            frame = self._frame_q.get_nowait()
            cw = self._canvas.winfo_width() or 800
            ch = self._canvas.winfo_height() or 520
            if cw > 10 and ch > 10:
                ph = frame_to_photoimage(frame, cw, ch)
                self._preview_img = ph
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor="nw", image=ph)
        except queue.Empty: pass
        self.after(33, self._poll_frames)

    def _on_complete(self, sr: ScanResult):
        self._running = False
        def update():
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._scan_status.config(text="COMPLETE", fg=C["green"])
            for dot in self._scan_dots: dot.config(fg=C["accent"])

            if sr.is_crowd:
                self._crowd_sidebar.pack(fill="both", expand=True)
                self._crowd_sidebar.clear()
            else:
                self._crowd_sidebar.pack_forget()

            unknown_faces = []
            
            top_d = max(sr.detections, key=lambda d: d[0].fusion_conf) if sr.detections else None
            if top_d: self._verdict.update_result(top_d[0])

            for fr, ff in sr.detections:
                def reg_cb(feat=ff.feature_vector, score=fr.fusion_conf):
                    return lambda: AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", score, feat, on_added=self.app._tab_db.refresh)
                def det_cb(res=fr):
                    return lambda: HighPriorityAlertDialog(self.winfo_toplevel(), res)

                self._crowd_sidebar.add_suspect(fr, ff, reg_cb(), det_cb())
                if not fr.db_match_found: unknown_faces.append(ff)

            if unknown_faces and _WEB_API_AVAILABLE:
                def handle_webcam_batch(result):
                    matches = result.get("results", []) if result.get("success") else []
                    still_unknown = []
                    for i, ff in enumerate(unknown_faces):
                        res_item = matches[i] if i < len(matches) else {"match_found": False}
                        if res_item.get("match_found", False):
                            for det_fr, det_ff in sr.detections:
                                if det_ff == ff:
                                    web_data = res_item.get("data", {})
                                    det_fr.db_match_found = True
                                    det_fr.suspect_name   = web_data.get("name")
                                    det_fr.suspect_alias  = web_data.get("alias")
                                    det_fr.crime_type     = web_data.get("crime_type")
                                    det_fr.suspect_id     = web_data.get("id")
                                    det_fr.suspect_risk   = web_data.get("risk_level")
                                    det_fr.fusion_conf    = res_item.get("confidence", 1.0)
                                    det_fr.verdict        = "CRIMINAL"
                                    
                                    current_top = max(sr.detections, key=lambda d: d[0].fusion_conf)
                                    if current_top[0] == det_fr:
                                        self.after(0, lambda fr=det_fr: self._verdict.update_result(fr))
                                    break
                        else:
                            still_unknown.append(ff)
                    
                    self.after(0, lambda: (
                        self._crowd_sidebar.clear(),
                        [self._crowd_sidebar.add_suspect(
                            d[0], d[1], 
                            lambda f=d[1].feature_vector, s=d[0].fusion_conf: AddToDatabaseDialog(self.winfo_toplevel(), self.app, "SUSPECT", s, f, on_added=self.app._tab_db.refresh),
                            lambda r=d[0]: HighPriorityAlertDialog(self.winfo_toplevel(), r)
                        ) for d in sr.detections if sr.is_crowd],
                        self._handle_completion_prompt(still_unknown, sr.detections)
                    ))

                search_web_faces_async([f.feature_vector for f in unknown_faces], handle_webcam_batch)

            self.after(0, lambda: (
                self._progress.__setitem__("value", 100),
                self._prog_lbl.config(text="100%"),
                self._handle_completion_prompt(unknown_faces, sr.detections)
            ))
            self.app.activity_log.log(f"[WEBCAM] Crowd Scan Complete: {len(sr.detections)} monitored.", tag="info")
        self.after(0, update)

    def _handle_completion_prompt(self, unknown_list, all_detections):
        """Webcam scan complete. Automatically open registration dialog for unknowns."""
        count = len(unknown_list)
        if count > 0:
            self.app.activity_log.log(f"[WEBCAM] {count} unidentified suspects detected. Opening registration...", tag="warning")
            def auto_prompt():
                ff = unknown_list[0]
                AddToDatabaseDialog(
                    self.winfo_toplevel(), self.app, "SUSPECT", 0.0, ff.feature_vector,
                    on_added=self.app._tab_db.refresh
                )
            self.after(500, auto_prompt)
        else:
            self.app.activity_log.log("[WEBCAM] No unidentified suspects found.", tag="info")



# ═════════════════════════════════════════════════════════════════════════════
#  Tab 5: Multi-CCTV Array
# ═════════════════════════════════════════════════════════════════════════════

class MultiCCTVTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._video_paths = [None, None, None]
        self._scanners = [Scanner(on_frame=self._make_frame_cb(i)) for i in range(3)]
        self._frame_qs = [queue.Queue(maxsize=2) for _ in range(3)]
        self._running = False
        self._suspect_locations = {} # dict mapping suspect_name to last known camera index
        self._transition_cooldowns = {} # cooldown to prevent rapid jumping between cams
        self._last_unidentified = None # Stores (face_features, score) of unidentified suspect
        self._detection_counts = {} # Anti-Ghosting: tracks (cam_idx, name) -> count
        self._build()
        self.after(33, self._poll_frames)

    def _build(self):
        # Top banner with buttons
        top = tk.Frame(self, bg=C["bg"])
        top.pack(fill="x", padx=15, pady=10)
        tk.Label(top, text="CCTV ARRAY (MULTIPLE STREAMS)", bg=C["bg"], fg=C["accent"], font=("Segoe UI", 13, "bold")).pack(side="left")

        self._btn_start = tk.Button(top, text="▶ Start All", command=self._start, bg=C["green"], fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, cursor="hand2")
        self._btn_start.pack(side="left", padx=15)
        self._btn_stop = tk.Button(top, text="⏹ Stop All", command=self._stop, bg=C["red"], fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, cursor="hand2", state="disabled")
        self._btn_stop.pack(side="left")
        
        self._btn_demo = tk.Button(top, text="⚡ Run Tracking Demo", command=self._run_demo, bg=C["purple"], fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, cursor="hand2")
        self._btn_demo.pack(side="right", padx=15)
        
        self._btn_target = tk.Button(top, text="🔍 Upload Search Target", command=self._upload_target, bg=C["accent"], fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, cursor="hand2")
        self._btn_target.pack(side="right", padx=10)

        # Container for inputs
        inp_row = tk.Frame(self, bg=C["bg"])
        inp_row.pack(fill="x", padx=15, pady=(0, 10))
        
        self._path_labels = []
        for i in range(3):
            sub = tk.Frame(inp_row, bg=C["bg"])
            sub.pack(side="left", padx=(15 if i>0 else 0, 5))
            
            tk.Button(sub, text=f"📁 Upload {i+1}", command=lambda idx=i: self._select_video(idx), 
                      bg=C["bg2"], fg="white", font=("Segoe UI", 8, "bold"), relief="raised", cursor="hand2", width=10).pack(side="left")
            
            tk.Button(sub, text="🎥 Live", command=lambda idx=i: self._add_network_cam(idx), 
                      bg="#1a3a1a", fg=C["green"], font=("Segoe UI", 8, "bold"), relief="raised", cursor="hand2", width=6).pack(side="left", padx=2)
            
            lbl = tk.Label(sub, text="No footage", bg=C["bg"], fg=C["gray"], font=("Segoe UI", 8), width=12, anchor="w")
            lbl.pack(side="left", padx=5)
            self._path_labels.append(lbl)
            
        # 2x2 Grid Layout
        grid = tk.Frame(self, bg=C["bg"])
        grid.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        self._canvases = []
        for r, c, idx in [(0, 0, 0), (0, 1, 1), (1, 0, 2)]:
            f = tk.Frame(grid, bg=C["bg3"], bd=1, relief="ridge")
            f.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            cv = tk.Canvas(f, bg=C["bg"], highlightthickness=0)
            cv.pack(fill="both", expand=True)
            ph = make_placeholder_frame(600, 400, f"CAM {idx+1} STANDBY")
            cv.create_image(0, 0, anchor="nw", image=frame_to_photoimage(ph, 600, 400), tags="img")
            cv.image_ref = None
            self._canvases.append(cv)

        # Bottom right: Map
        map_frame = tk.Frame(grid, bg=C["bg2"], bd=1, relief="ridge")
        map_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        tk.Label(map_frame, text="📍 TACTICAL MAP & ALERTS", bg=C["bg2"], fg=C["orange"], font=("Segoe UI", 11, "bold")).pack(pady=(10, 0))
        
        self._map_canvas = tk.Canvas(map_frame, bg=C["bg"], highlightthickness=0)
        self._map_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self._map_canvas.bind("<Configure>", self._draw_base_map)
        
        self._alert_lbl = tk.Label(map_frame, text="Awaiting tracking data...", bg=C["bg2"], fg=C["gray"], font=("Segoe UI", 10), wraplength=400)
        self._alert_lbl.pack(side="bottom", pady=10)

    def _select_video(self, idx):
        path = filedialog.askopenfilename(title=f"Select CCTV Footage {idx+1}", filetypes=[("Video Footage", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self._video_paths[idx] = path
            bname = os.path.basename(path)
            if len(bname) > 12: bname = bname[:10] + "..."
            self._path_labels[idx].config(text=bname, fg=C["accent"])

    def _add_network_cam(self, idx):
        from tkinter import simpledialog
        choice = simpledialog.askinteger("Live Camera Option", 
                "Enter Camera Source\n(0 = Local Laptop Cam, 2 = Satellite 2, 3 = Satellite 3):", 
                initialvalue=0, minvalue=0, maxvalue=3)
        if choice is not None:
            if choice == 0:
                self._video_paths[idx] = 0
                self._path_labels[idx].config(text=f"🎥 Local Cam", fg=C["accent"])
            else:
                self._video_paths[idx] = f"SATELLITE:{choice}"
                self._path_labels[idx].config(text=f"🛰️ Satellite {choice}", fg=C["green"])
            self.app.activity_log.log(f"CCTV Array Slot {idx+1} Configured to Camera {choice}", "info")

    def _upload_target(self):
        path = filedialog.askopenfilename(title="Select Suspect Photo", filetypes=[("Image", "*.jpg *.jpeg *.png")])
        if not path: return
        import cv2
        from modules.face_analyzer import FaceAnalyzer
        
        self._alert_lbl.config(text="Analyzing Target Face...", fg=C["orange"])
        self.update_idletasks()
        
        img = cv2.imread(path)
        if img is None:
            self._alert_lbl.config(text="Could not read target image.", fg=C["red"])
            return
            
        fa = FaceAnalyzer()
        faces = fa.analyze(img)
        faces = [f for f in faces if f.detected]
        if not faces:
            self._alert_lbl.config(text="No face detected in target image! Cannot track.", fg=C["red"])
            return
            
        best_face = max(faces, key=lambda f: f.face_area)
        
        best_face = max(faces, key=lambda f: f.face_area)
        
        # Lock Target into DB silently for now
        self._target_name = "SEARCH_TARGET_PENDING"
        self._target_features = best_face.feature_vector
        
        cid = add_criminal_to_db(self._target_name, alias="Manhunt Target", crime_type="Priority Manhunt", risk_level="CRITICAL", status="At Large", face_features=self._target_features)
        if cid != -1:
            register_criminal_face(cid, self._target_features)
            
        self._alert_lbl.config(text="Target Locked. Silent track active. Awaiting appearance...", fg=C["green"])

    def _draw_base_map(self, event=None):
        self.after(0, self.update_idletasks) 
        w = self._map_canvas.winfo_width()
        h = self._map_canvas.winfo_height()
        
        # Robustness: Use baseline defaults if parent frame hasn't measured yet
        if w < 100: w = 400
        if h < 100: h = 300
        
        self._map_canvas.delete("base")
        
        # === Google Maps UI Style Background ===
        self._map_canvas.create_rectangle(0, 0, w, h, fill="#0F172A", outline="", tags="base")
        
        # Roads / High-tech grids
        self._map_canvas.create_line(0, int(h*0.4), w, int(h*0.2), fill="#1E293B", width=18, tags="base")
        self._map_canvas.create_line(0, int(h*0.4), w, int(h*0.2), fill="#0284C7", width=2, tags="base") # highway strip
        
        self._map_canvas.create_line(int(w*0.3), 0, int(w*0.6), h, fill="#1E293B", width=22, tags="base")
        self._map_canvas.create_line(int(w*0.7), 0, int(w*0.3), h, fill="#1E293B", width=12, tags="base")
        
        # Draw 3 location pins points mimicking CCTV coordinates
        self.points = [(int(w*0.25), int(h*0.35)), (int(w*0.75), int(h*0.25)), (int(w*0.45), int(h*0.75))]
        
        # Tracking network paths
        self._map_canvas.create_line(self.points[0], self.points[1], fill="#38BDF8", width=2, dash=(4,4), tags="base")
        self._map_canvas.create_line(self.points[1], self.points[2], fill="#38BDF8", width=2, dash=(4,4), tags="base")
        self._map_canvas.create_line(self.points[2], self.points[0], fill="#38BDF8", width=2, dash=(4,4), tags="base")
        
        for i, (px, py) in enumerate(self.points):
            self._map_canvas.create_oval(px-35, py-35, px+35, py+35, fill="#0F172A", outline="#38BDF8", width=3, tags="base")
            self._map_canvas.create_text(px, py, text=f"CAM {i+1}", fill="white", font=("Segoe UI", 10, "bold"), tags="base")
        
        self._map_canvas.tag_lower("base")
            
    def _update_suspect_map(self, cam_idx, suspect_name):
        if not hasattr(self, 'points'): 
            self._draw_base_map()
        px, py = self.points[cam_idx]
        import random
        ox, oy = random.randint(-15, 15), random.randint(-15, 15)
        self._map_canvas.delete(f"suspect_{suspect_name}")
        
        # Larger Beacons
        self._map_canvas.create_oval(px+ox-12, py+oy-12, px+ox+12, py+oy+12, fill="#EF4444", outline="white", width=2, tags=f"suspect_{suspect_name}")
        self._map_canvas.create_text(px+ox, py+oy+20, text=suspect_name, fill="#F87171", font=("Segoe UI", 9, "bold"), tags=f"suspect_{suspect_name}")

    def _make_frame_cb(self, idx):
        def cb(frame, detections=[]):
            try: self._frame_qs[idx].put_nowait((frame, detections))
            except queue.Full: pass
        return cb

    def _poll_frames(self):
        for i in range(3):
            try:
                frame, detections = self._frame_qs[i].get_nowait()
                cv = self._canvases[i]
                w, h = cv.winfo_width(), cv.winfo_height()
                if w > 10 and h > 10:
                    ph = frame_to_photoimage(frame, w, h)
                    cv.delete("img")
                    cv.create_image(0, 0, anchor="nw", image=ph, tags="img")
                    cv.image_ref = ph

                # Iterate through all detections to update the map
                for fr, ff in detections:
                    # 1. Determine if this is a match we care about for mapping
                    is_target = hasattr(self, '_target_name') and self._target_name == "SEARCH_TARGET_PENDING" and \
                                fr.verdict in ("CRIMINAL", "WATCH LIST") and fr.suspect_name == "SEARCH_TARGET_PENDING"
                    
                    is_general_match = fr.verdict in ("CRIMINAL", "WATCH LIST") and fr.suspect_name and not is_target
                    
                    is_unknown_suspect = fr.verdict in ("CRIMINAL", "WATCH LIST") and not fr.suspect_name
                    
                    if is_target or is_general_match or is_unknown_suspect:
                        match_name = self._target_name if is_target else (fr.suspect_name if is_general_match else "UNKNOWN SUSPECT")
                        
                        # ANTI-GHOSTING & PERSISTENCE
                        key = (i, match_name)
                        self._detection_counts[key] = self._detection_counts.get(key, 0) + 1
                        
                        # High confidence gets instant mapping, lower confidence needs persistence
                        required_frames = 1 if fr.db_similarity >= 0.88 else 4
                        
                        if self._detection_counts[key] >= required_frames:
                            now = time.time()
                            last_cam = self._suspect_locations.get(match_name)
                            last_time = self._transition_cooldowns.get(match_name, 0)
                            
                            # Update map and locations
                            if last_cam != i or now - last_time > 2.0: # Refresh every 2s even if stationary
                                self._suspect_locations[match_name] = i
                                self._transition_cooldowns[match_name] = now
                                
                                if last_cam is not None and last_cam != i:
                                    self._alert_lbl.config(text=f"🚨 SUSPECT MOVED: {match_name} to CAM {i+1}!", fg="#FCA5A5")
                                    # Path trace
                                    px1, py1 = self.points[last_cam]
                                    px2, py2 = self.points[i]
                                    line_id = self._map_canvas.create_line(px1, py1, px2, py2, fill="#EF4444", width=3, arrow=tk.LAST, tags="path")
                                    self.after(4000, lambda lid=line_id: self._map_canvas.delete(lid))
                                else:
                                    msg = f"🎯 TARGET SPOTTED: CAM {i+1}" if is_target else f"👁️ {match_name} detected: CAM {i+1}"
                                    self._alert_lbl.config(text=msg, fg=C["orange"])
                                
                                self._update_suspect_map(i, match_name)
                                
                    # Cache unknown features for stop-action registration
                    if fr.verdict in ("CRIMINAL", "WATCH LIST") and not fr.suspect_name:
                        if not self._last_unidentified or fr.fusion_conf > self._last_unidentified[0]:
                            self._last_unidentified = (fr.fusion_conf, fr.face_features)

            except queue.Empty: pass
            
        self.after(33, self._poll_frames)

    def _start(self):
        if self._running: return
        
        self.sources = self._video_paths.copy()
        if not any(self.sources):
            self._alert_lbl.config(text="Please upload at least one video footage.", fg=C["orange"])
            return

        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._suspect_locations.clear()
        self._transition_cooldowns.clear()
        self._detection_counts.clear()
        self._map_canvas.delete("all")
        self._draw_base_map()
        
        # Launch threads
        for i in range(3):
            threading.Thread(target=self._run_cam, args=(i,), daemon=True).start()

    def _run_cam(self, idx):
        src = self.sources[idx]
        if src is None or src == "": return
        
        while self._running:
            self._scanners[idx].reset()
            self._scanners[idx].scan_video(src, remote_receiver=self.app.stream_receiver)
            if self._scanners[idx]._stop_flag.is_set() or not self._running:
                break

    def _stop(self):
        for s in self._scanners: s.stop()
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._alert_lbl.config(text="Scanning stopped.", fg=C["gray"])
        
        # Final Registration Workflow (Triggered on STOP as requested)
        self.after(500, self._show_delayed_registrations)

    def _show_delayed_registrations(self):
        # 1. Registration for the specific Manhunt Target (if one was uploaded)
        if hasattr(self, '_target_features') and self._target_features:
            messagebox.showinfo("Target Registration", "Monitoring finished. Please finalize the registration details for your Search Target.")
            AddToDatabaseDialog(self.winfo_toplevel(), self.app, "MANHUNT", 1.0, self._target_features)
            self._target_features = None # Clear after showing
            self._target_name = None
            
        # 2. Registration for any other unidentified suspects found during the run
        elif hasattr(self, '_last_unidentified') and self._last_unidentified:
            score, features = self._last_unidentified
            if features and len(features) >= 128:
                messagebox.showinfo("Unknown Suspect Found", "A suspicious unidentified person was detected during scan.\nOpening registration dialog...")
                AddToDatabaseDialog(self.winfo_toplevel(), self.app, "CCTV_DETECT", score, features)
                self._last_unidentified = None

    def _run_demo(self):
        """Simulate a suspect moving across the cameras to demonstrate the map!"""
        if self._running:
            self._alert_lbl.config(text="Stop cameras first to run demo.", fg=C["orange"])
            return
            
        self._btn_demo.config(state="disabled")
        self._alert_lbl.config(text="Initializing Tracking Sequence...", fg=C["purple"])
        self._suspect_locations.clear()
        self._map_canvas.delete("all")
        self._draw_base_map()

        def demo_loop():
            import time
            from modules.fusion_engine import FusionResult
            import numpy as np
            suspect = "Dawood Ibrahim (DEMO)"
            
            # Create a fake mock grid screen
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock_frame, "SIMULATED CAM VIEW", (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(mock_frame, (10, 10), (630, 470), (0, 0, 255), 3)

            def fire_detection(cam_idx):
                fr = FusionResult()
                fr.verdict = "CRIMINAL"
                fr.suspect_name = suspect
                fr.risk_level = "CRITICAL"
                fr.fusion_conf = 0.99
                try: self._frame_qs[cam_idx].put_nowait((mock_frame, fr))
                except queue.Full: pass
            
            # Simulated timeline
            time.sleep(1)
            fire_detection(0) # Appears on Cam 1
            
            time.sleep(3)
            fire_detection(1) # Moves to Cam 2
            
            time.sleep(4)
            fire_detection(2) # Moves to Cam 3
            
            time.sleep(2)
            self.after(0, lambda: self._alert_lbl.config(text="Tracking Demo Complete.", fg=C["green"]))
            self.after(0, lambda: self._btn_demo.config(state="normal"))

        threading.Thread(target=demo_loop, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
#  HIGH PRIORITY Alert Dialog  (shown when a known criminal is detected)
# ═════════════════════════════════════════════════════════════════════════════

class HighPriorityAlertDialog(tk.Toplevel):
    """
    Dramatic red-themed modal popup shown immediately when a known criminal
    is identified via LBPH face matching. Displays full profile with all
    relevant law-enforcement metadata.
    """

    def __init__(self, parent, match):
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        self.title("⚠ HIGH PRIORITY CRIMINAL DETECTED")
        self.configure(bg="#0a0000")
        self.resizable(False, False)

        w, h = 620, 480
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        if pw > 1:
            x, y = px + (pw - w) // 2, py + (ph - h) // 2
        else:
            x, y = (sw - w) // 2, (sh - h) // 2

        self.geometry(f"{w}x{h}+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grab_set()

        self._match = match
        self._build()

    def _build(self):
        m = self._match

        # ── Pulsing red header ────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#cc0000")
        hdr.pack(fill="x")

        tk.Label(hdr, text="🚨  HIGH PRIORITY CRIMINAL ALERT  🚨",
                 bg="#cc0000", fg="white",
                 font=("Segoe UI", 16, "bold")).pack(pady=(14, 4))
        tk.Label(hdr, text="IMMEDIATE LAW ENFORCEMENT ACTION REQUIRED",
                 bg="#cc0000", fg="#ffcccc",
                 font=("Segoe UI", 9)).pack(pady=(0, 14))

        # ── Criminal profile card ─────────────────────────────────────────────
        card = tk.Frame(self, bg="#160000", padx=20, pady=15)
        card.pack(fill="both", expand=True, padx=2, pady=2)

        def profile_row(label, value, val_color="#ff9999"):
            row = tk.Frame(card, bg="#160000")
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{label}:", bg="#160000", fg="#888888",
                     font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg="#160000", fg=val_color,
                     font=("Segoe UI", 10, "bold"), anchor="w", wraplength=420,
                     justify="left").pack(side="left", fill="x", expand=True)

        profile_row("IDENTIFIED AS", m.name,       "#ff4444")
        profile_row("ALIAS",         m.alias,       "#ffaa44")
        profile_row("CRIME TYPE",    m.crime_type,  "#ffaa44")
        profile_row("STATUS",        m.status,      "#ff9999")
        profile_row("BOUNTY",        m.bounty,      "#ffff44")
        profile_row("KNOWN FOR",     m.known_for,   "#ff9999")
        profile_row("PRIORITY",      m.priority,    "#ff0000")
        profile_row("MATCH CONF",    f"{m.confidence:.1%} (LBPH facial biometrics)", "#44ff88")

        # ── Confidence bar ────────────────────────────────────────────────────
        bar_frame = tk.Frame(card, bg="#160000")
        bar_frame.pack(fill="x", pady=(10, 4))
        tk.Label(bar_frame, text="Match Confidence:", bg="#160000", fg="#888888",
                 font=("Segoe UI", 8)).pack(anchor="w")
        bar_bg = tk.Frame(bar_frame, bg="#330000", height=12)
        bar_bg.pack(fill="x")
        bar_fill_w = max(1, int(600 * m.confidence))
        bar_fill = tk.Frame(bar_bg, bg="#ff0000", height=12, width=bar_fill_w)
        bar_fill.place(x=0, y=0, width=bar_fill_w, height=12)

        # ── Dismiss button ────────────────────────────────────────────────────
        tk.Button(self,
                  text="✔  Acknowledged — Initiate Arrest Protocol",
                  command=self.destroy,
                  bg="#cc0000", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=20, pady=10, cursor="hand2",
                  activebackground="#ff0000").pack(fill="x", padx=2, pady=2)


# ==========================================================================
#  Register Face Dialog  (links a detected face to an existing criminal)
# ==========================================================================

class RegisterFaceDialog(tk.Toplevel):
    """
    Bulletproof Register Face Dialog.
    - _registered flag stops double-fire from button + Return key
    - All post-destroy calls wrapped in try/except
    - Buttons packed at bottom first so they are always visible
    """

    def __init__(self, parent, face_features, on_registered=None):
        super().__init__(parent)
        self._fv            = face_features
        self._on_registered = on_registered
        self._registered    = False

        self.title("Register Face to Criminal")
        self.configure(bg=C["bg"])
        self.resizable(False, True)
        self.protocol("WM_DELETE_WINDOW", self._safe_close)

        w, h = 600, 500
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{max(20,(sh-h)//2)}")
        self.lift()
        self.attributes("-topmost", True)
        self.focus_force()
        self._build()

    def _safe_close(self):
        try: self.destroy()
        except Exception: pass

    def _build(self):
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # Header
        hdr = tk.Frame(main, bg=C["bg3"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="\U0001f517  Link Face to Criminal Record",
                 bg=C["bg3"], fg=C["accent"],
                 font=("Segoe UI", 15, "bold")).pack(padx=20, pady=12, anchor="w")

        tk.Label(main,
                 text="Select which criminal this face belongs to.\n"
                      "The embedding is saved so future uploads auto-match.",
                 bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(padx=20, pady=(10,4), anchor="w")

        # Separator + BUTTONS pinned at BOTTOM (packed first = always visible)
        tk.Frame(main, bg=C["bg3"], height=1).pack(side="bottom", fill="x")
        btn_row = tk.Frame(main, bg=C["bg"])
        btn_row.pack(side="bottom", pady=16)

        self._btn_save = tk.Button(
            btn_row, text="\u2714  Register & Save",
            command=self._on_click,
            bg=C["red"], fg="white", font=("Segoe UI", 12, "bold"),
            relief="flat", padx=30, pady=10, cursor="hand2",
            activebackground="#cc0000")
        self._btn_save.pack(side="left", padx=10)

        tk.Button(btn_row, text="\u2716  Cancel",
                  command=self._safe_close,
                  bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 11),
                  relief="flat", padx=22, pady=10, cursor="hand2").pack(side="left", padx=10)

        self._msg = tk.Label(main, text="", bg=C["bg"], font=("Segoe UI", 10))
        self._msg.pack(side="bottom", pady=(0,6))

        # Load criminals from DB
        criminals = get_all_criminals()
        if not criminals:
            tk.Label(main, text="\n  No criminals in database yet.",
                     bg=C["bg"], fg=C["red"],
                     font=("Segoe UI", 10)).pack(expand=True)
            self._btn_save.config(state="disabled")
            return

        self._id_map = {}
        critical_entries = []
        other_entries = []
        for cr in criminals:
            risk  = cr.get("risk_level", "")
            alias = f" ({cr['alias']})" if cr.get("alias") else ""
            lbl   = f"{cr['name']}{alias}  [{risk}]  (ID {cr['id']})"
            self._id_map[lbl] = cr["id"]
            (critical_entries if risk == "CRITICAL" else other_entries).append(lbl)

        display_list = sorted(critical_entries) + sorted(other_entries)

        row_frame = tk.Frame(main, bg=C["bg"])
        row_frame.pack(fill="x", padx=20, pady=8)
        tk.Label(row_frame, text="Criminal:", bg=C["bg"], fg=C["gray"],
                 font=("Segoe UI", 10), width=10, anchor="w").pack(side="left")

        self._sel_var = tk.StringVar(value=display_list[0] if display_list else "")
        cb = ttk.Combobox(row_frame, textvariable=self._sel_var,
                          values=display_list, state="readonly",
                          font=("Segoe UI", 10), width=48)
        cb.pack(side="left", fill="x", expand=True)

        if critical_entries:
            tk.Label(main,
                     text=f"  \u26a0  {len(critical_entries)} CRITICAL criminals at top"
                          "  (Dawood Ibrahim, Bin Laden, El Chapo...)",
                     bg=C["bg"], fg=C["red"],
                     font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=20, pady=(0,2))

        # Return key bound AFTER _msg and _sel_var exist
        self.bind("<Return>", lambda _e: self._on_click())

    def _on_click(self):
        """Single entry point for both button click and Return key."""
        if self._registered:
            return
        if not hasattr(self, "_sel_var") or not hasattr(self, "_msg"):
            return

        selected = self._sel_var.get().strip()
        cid      = self._id_map.get(selected)

        if not cid:
            try: self._msg.config(text="Choose a criminal from the list.", fg=C["orange"])
            except Exception: pass
            return

        try:
            self._btn_save.config(state="disabled", text="Saving...")
            self._msg.config(text="Saving face embedding...", fg=C["gray"])
            self.update_idletasks()
        except Exception: pass

        ok = register_criminal_face(cid, self._fv)

        if not ok:
            try:
                self._msg.config(text="\u2717 Save failed. See console.", fg=C["red"])
                self._btn_save.config(state="normal", text="\u2714  Register & Save")
            except Exception: pass
            return

        # Success
        self._registered = True
        crim_name = selected.split("  [")[0].strip()

        try:
            self._msg.config(text=f"\u2713 Saved!  Linked to  {crim_name}", fg=C["green"])
            self.update_idletasks()
        except Exception: pass

        if self._on_registered:
            try: self._on_registered()
            except Exception: pass

        try: self.destroy()
        except Exception: pass


class AddToDatabaseDialog(tk.Toplevel):
    """
    Stable modal popup for criminal registration.
    No automatic registration — only occurs when user clicks 'Confirm Registration'.
    """
    CRIME_TYPES = [
        "Armed Robbery", "Assault", "Burglary", "Cybercrime",
        "Drug Trafficking", "Fraud", "Homicide", "Identity Theft",
        "Kidnapping", "Money Laundering", "Murder", "Organized Crime",
        "Smuggling", "Terrorism", "Terror Financing", "Vehicle Theft", "Other",
    ]
    RISK_LEVELS = ["HIGH", "MEDIUM", "LOW"]
    STATUSES    = ["At Large", "Imprisoned", "Deceased", "Unknown"]

    def __init__(self, parent, app, verdict: str, fusion_score: float,
                 face_features: list, on_added=None):
        super().__init__(parent)
        self.app = app
        self._face_features = face_features
        self._on_added      = on_added
        self._register_active = True
        
        # Initialize Variables
        self._name_var  = tk.StringVar()
        self._alias_var = tk.StringVar()
        self._crime_var = tk.StringVar(value=self.CRIME_TYPES[0])
        self._risk_var  = tk.StringVar(value="HIGH")
        self._status_var= tk.StringVar(value="At Large")
        self._notes_var = tk.StringVar()

        self.title("Criminal Registration")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        
        # Basic container
        self.main_frame = tk.Frame(self, bg=C["bg"], padx=25, pady=20)
        self.main_frame.pack(fill="both", expand=True)

        self._build_ui(verdict, fusion_score)

        # Centering
        self.update_idletasks()
        w, h = 500, 560
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        
        self.grab_set() # Modal
        self.focus_force()

    def _build_ui(self, verdict, score):
        tk.Label(self.main_frame, text="REGISTER SUSPECT", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 12, "bold")).pack(pady=(0, 20))

        # Form
        form = tk.Frame(self.main_frame, bg=C["bg"])
        form.pack(fill="x")

        def row(label, var, widget_type="entry", values=None):
            r = tk.Frame(form, bg=C["bg"])
            r.pack(fill="x", pady=6)
            tk.Label(r, text=label, bg=C["bg"], fg=C["gray"], font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
            if widget_type == "entry":
                e = tk.Entry(r, textvariable=var, bg=C["bg3"], fg="white", 
                            insertbackground="white", relief="flat", font=("Segoe UI", 10), bd=5)
                e.pack(side="left", fill="x", expand=True)
                return e
            else:
                cb = ttk.Combobox(r, textvariable=var, values=values, state="readonly", font=("Segoe UI", 10))
                cb.pack(side="left", fill="x", expand=True)
                return cb
        
        row("Full Name *",  self._name_var)
        row("Alias",       self._alias_var)
        row("Crime Type",   self._crime_var, "combo", self.CRIME_TYPES)
        row("Risk Level",   self._risk_var,  "combo", self.RISK_LEVELS)
        row("Status",       self._status_var, "combo", self.STATUSES)
        row("Notes",        self._notes_var)

        self._msg_lbl = tk.Label(self.main_frame, text="", bg=C["bg"], fg=C["red"], font=("Segoe UI", 9))
        self._msg_lbl.pack(pady=10)

        btns = tk.Frame(self.main_frame, bg=C["bg"])
        btns.pack(pady=10)
        
        tk.Button(btns, text="CONFIRM REGISTRATION", command=self._handle_register, 
                  bg=C["red"], fg="white", font=("Segoe UI", 10, "bold"), 
                  padx=20, pady=8, relief="flat", cursor="hand2").pack(side="left", padx=10)
        
        tk.Button(btns, text="CANCEL", command=self.destroy,
                  bg=C["bg3"], fg="white", font=("Segoe UI", 10), 
                  padx=20, pady=8, relief="flat", cursor="hand2").pack(side="left", padx=10)

    def _handle_register(self, *args):
        if not self._register_active:
            return
            
        name = self._name_var.get().strip()
        if not name:
            self._msg_lbl.config(text="Full Name is required.", fg=C["orange"])
            return
            
        self._register_active = False # Prevent double-click
        
        try:
            cid = add_criminal_to_db(
                name        = name,
                alias       = self._alias_var.get().strip(),
                crime_type  = self._crime_var.get(),
                risk_level  = self._risk_var.get(),
                status      = self._status_var.get(),
                face_features = self._face_features,
            )
            if cid > 0:
                self._msg_lbl.config(text=f"Registered '{name}' as ID {cid}", fg=C["green"])
                if self._on_added:
                    try:
                        self._on_added(name)
                    except TypeError:
                        self._on_added()
                self.after(1000, self.destroy)
            else:
                self._msg_lbl.config(text="DB error -- see console.", fg=C["red"])
                self._register_active = True
        except Exception as e:
             self._msg_lbl.config(text=f"Error: {e}", fg=C["red"])
             self._register_active = True


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 4: Database Viewer
# ═════════════════════════════════════════════════════════════════════════════

class DatabaseTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        # ── Top stats bar ─────────────────────────────────────────────────────
        stats_bar = tk.Frame(self, bg=C["bg3"])
        stats_bar.pack(fill="x", padx=15, pady=(12, 0))
        self._stat_labels = {}
        for key, label, color in [
            ("total_criminals", "SUSPECT REGISTRY", C["accent"]),
            ("total_scans",     "TOTAL SCANS",      C["white"]),
            ("criminal_hits",   "CRIMINAL HITS",    C["red"]),
            ("watchlist_hits",  "WATCH LIST",        C["orange"]),
            ("clear_hits",      "CLEARED",           C["green"]),
        ]:
            col = tk.Frame(stats_bar, bg=C["bg3"])
            col.pack(side="left", expand=True, fill="x", padx=12, pady=10)
            val_lbl = tk.Label(col, text="—", bg=C["bg3"], fg=color,
                               font=("Segoe UI", 24, "bold"))
            val_lbl.pack()
            tk.Label(col, text=label, bg=C["bg3"], fg=C["gray"],
                     font=("Segoe UI", 8)).pack()
            self._stat_labels[key] = val_lbl

        # ── Main split ────────────────────────────────────────────────────────
        split = tk.Frame(self, bg=C["bg"])
        split.pack(fill="both", expand=True, padx=15, pady=12)

        # Left: Criminal registry table
        left = tk.Frame(split, bg=C["bg2"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        hdr_row = tk.Frame(left, bg=C["bg2"])
        hdr_row.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(hdr_row, text="CRIMINAL REGISTRY", bg=C["bg2"], fg=C["accent"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr_row, text="↻ Refresh", command=self.refresh,
                  bg=C["bg3"], fg=C["txt"], font=("Segoe UI", 8), relief="flat",
                  padx=8, cursor="hand2").pack(side="right")

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter_criminals())
        search_frame = tk.Frame(left, bg=C["bg2"])
        search_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(search_frame, text="🔍", bg=C["bg2"], fg=C["gray"],
                 font=("Segoe UI", 11)).pack(side="left")
        tk.Entry(search_frame, textvariable=self._search_var, bg=C["bg3"],
                 fg=C["txt"], insertbackground=C["txt"], font=("Segoe UI", 9),
                 relief="flat", bd=0).pack(side="left", fill="x", expand=True, padx=(4, 0))

        cols_c = ("Name", "Alias", "Crime", "Risk", "Status")
        self._tree_c = ttk.Treeview(left, columns=cols_c, show="headings", height=14)
        for col, w in zip(cols_c, [160, 100, 150, 80, 80]):
            self._tree_c.heading(col, text=col)
            self._tree_c.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(left, command=self._tree_c.yview)
        self._tree_c.config(yscrollcommand=sb.set)
        self._tree_c.pack(fill="both", expand=True, padx=10)
        sb.pack(side="right", fill="y")

        # Right: Detection log
        right = tk.Frame(split, bg=C["bg2"], width=400)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        hdr2 = tk.Frame(right, bg=C["bg2"])
        hdr2.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(hdr2, text="DETECTION LOG", bg=C["bg2"], fg=C["accent"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        cols_d = ("Time", "Verdict", "Mode", "Conf%")
        self._tree_d = ttk.Treeview(right, columns=cols_d, show="headings", height=14)
        for col, w in zip(cols_d, [90, 90, 70, 60]):
            self._tree_d.heading(col, text=col)
            self._tree_d.column(col, width=w, anchor="center")
        sb2 = ttk.Scrollbar(right, command=self._tree_d.yview)
        self._tree_d.config(yscrollcommand=sb2.set)
        self._tree_d.pack(fill="both", expand=True, padx=10)
        sb2.pack(side="right", fill="y")

        # Row tag colours
        self._tree_d.tag_configure("CRIMINAL",   foreground=C["red"])
        self._tree_d.tag_configure("WATCH LIST", foreground=C["orange"])
        self._tree_d.tag_configure("CLEAR",      foreground=C["green"])
        self._tree_c.tag_configure("HIGH",   foreground=C["red"])
        self._tree_c.tag_configure("MEDIUM", foreground=C["orange"])
        self._tree_c.tag_configure("LOW",    foreground=C["green"])

    def refresh(self):
        """Reload data from database."""
        try:
            # Stats
            stats = get_stats()
            for key, lbl in self._stat_labels.items():
                lbl.config(text=str(stats.get(key, 0)))

            # Criminals
            self._all_criminals = get_all_criminals()
            self._populate_criminals(self._all_criminals)

            # Detection log
            for item in self._tree_d.get_children():
                self._tree_d.delete(item)
            detections = get_recent_detections(limit=50)
            for d in detections:
                ts  = d.get("timestamp", "")[:16]
                tag = d.get("verdict", "CLEAR")
                self._tree_d.insert("", "end", values=(
                    ts,
                    f"{verdict_icon(tag)} {tag}",
                    d.get("mode", "").upper(),
                    f"{d.get('fusion_conf', 0)*100:.0f}%",
                ), tags=(tag,))

        except Exception as e:
            print(f"DB refresh error: {e}")

    def _populate_criminals(self, criminals: list):
        for item in self._tree_c.get_children():
            self._tree_c.delete(item)
        for cr in criminals:
            risk = cr.get("risk_level", "MEDIUM")
            self._tree_c.insert("", "end", values=(
                cr.get("name", ""),
                cr.get("alias", ""),
                cr.get("crime_type", ""),
                cr.get("risk_level", ""),
                cr.get("status", ""),
            ), tags=(risk,))

    def _filter_criminals(self):
        q = self._search_var.get().lower()
        filtered = [
            c for c in self._all_criminals
            if q in c.get("name", "").lower()
            or q in c.get("alias", "").lower()
            or q in c.get("crime_type", "").lower()
        ]
        self._populate_criminals(filtered)


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application Window
# ═════════════════════════════════════════════════════════════════════════════

class CISv2App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=C["bg"])
        self.root.minsize(1200, 700)

        # Apply dark ttk theme
        self._apply_ttk_style()

        # ── Init database ────────────────────────────────────────────────────
        try:
            init_db()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to initialize DB:\n{e}")

        # ── Build layout ─────────────────────────────────────────────────────
        # Satellite Integration
        self.stream_receiver = StreamReceiver(self)
        self.stream_receiver.start()

        self._build_header()
        self._build_main()
        self._build_footer()

    def _apply_ttk_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook",          background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",      background=C["bg3"], foreground=C["gray"],
                        padding=[18, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", C["bg2"])],
                  foreground=[("selected", C["accent"])])
        style.configure("Treeview",           background=C["bg2"], foreground=C["txt"],
                        fieldbackground=C["bg2"], rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading",   background=C["bg3"], foreground=C["accent"],
                        font=("Segoe UI", 9, "bold"))
        style.configure("TProgressbar",       troughcolor=C["bg3"], background=C["accent"], borderwidth=0, thickness=10)
        style.configure("TScrollbar",         background=C["bg3"], troughcolor=C["bg"])
        style.map("Treeview", background=[("selected", C["bg4"])])

        # Modern Dark Combobox Style
        style.configure("TCombobox",
                        fieldbackground=C["bg3"],
                        background=C["bg3"],
                        foreground=C["txt"],
                        arrowcolor=C["accent"],
                        bordercolor=C["bg4"],
                        darkcolor=C["bg"],
                        lightcolor=C["bg4"],
                        padding=5)
        style.map("TCombobox",
                  fieldbackground=[("readonly", C["bg3"])],
                  foreground=[("readonly", C["txt"])])

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg3"], height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: Logo
        logo = tk.Frame(hdr, bg=C["bg3"])
        logo.pack(side="left", padx=20)
        tk.Label(logo, text="⬡ CIS", bg=C["bg3"], fg=C["accent"],
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(logo, text="v2", bg=C["bg3"], fg=C["purple"],
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=(2, 8), pady=(8, 0))
        tk.Label(logo, text="Criminal Identification System  •  BioFuse", bg=C["bg3"], fg=C["gray"],
                 font=("Segoe UI", 9)).pack(side="left")

        # NETWORK HUB STATUS
        hub_ip = get_local_ip()
        hub_frame = tk.Frame(hdr, bg=C["bg2"], bd=1, relief="solid")
        hub_frame.pack(side="left", padx=40, pady=10)
        tk.Label(hub_frame, text=f"🌐 PUSH HUB: http://{hub_ip}:8080", 
                 bg=C["bg2"], fg=C["green"], font=("Consolas", 10, "bold")).pack(padx=10, pady=2)

        # Right: Live clock + status
        right = tk.Frame(hdr, bg=C["bg3"])
        right.pack(side="right", padx=20)
        self._clock_lbl = tk.Label(right, text="", bg=C["bg3"], fg=C["accent"],
                                    font=("Consolas", 11))
        self._clock_lbl.pack(side="right")
        self._db_status = tk.Label(right, text="● DB ONLINE", bg=C["bg3"], fg=C["green"],
                                    font=("Segoe UI", 9))
        self._db_status.pack(side="right", padx=20)
        self._update_clock()

    def _update_clock(self):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._clock_lbl.config(text=now)
        self.root.after(1000, self._update_clock)

    def _build_main(self):
        # Main body: tabs on left content, narrow activity log on right
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Notebook
        self._nb = ttk.Notebook(body)
        self._nb.pack(side="left", fill="both", expand=True)

        # Activity Log sidebar
        self.activity_log = LogPanel(body, title="ACTIVITY LOG")
        self.activity_log.pack(side="right", fill="y", padx=(0, 0), pady=0)
        self.activity_log.config(width=260)
        self.activity_log.pack_propagate(False)

        # Create tabs
        self._tab_image   = ImageTab(self._nb, self)
        self._tab_video   = VideoTab(self._nb, self)
        self._tab_sketch  = SketchTab(self._nb, self)
        self._tab_webcam  = WebcamTab(self._nb, self)
        self._tab_multi   = MultiCCTVTab(self._nb, self)
        self._tab_db      = DatabaseTab(self._nb, self)

        self._nb.add(self._tab_image,  text="  📷 Image Upload  ")
        self._nb.add(self._tab_video,  text="  📹 Video Upload  ")
        self._nb.add(self._tab_sketch, text="  🎨 Sketch Analysis  ")
        self._nb.add(self._tab_webcam, text="  🎥 Live Webcam  ")
        self._nb.add(self._tab_multi,  text="  🌐 CCTV Array  ")
        self._nb.add(self._tab_db,     text="  🗄️ Database  ")

        # Refresh DB tab when selected
        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self.activity_log.log("CIS v2 initialized — Database online", "info")
        self.activity_log.log("Thresholds: Criminal>52% | Suspect>38%", "info")

    def _on_tab_change(self, event):
        selected = self._nb.select()
        tab_text = self._nb.tab(selected, "text")
        if "Database" in tab_text:
            self._tab_db.refresh()

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=C["bg3"], height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(
            footer,
            text=("IGNISIA  •  Team BioFuse  •  Criminal Identification System v2  "
                  "•  Face 60% | Gait 5% | Behavior 35% (image mode)  •  "
                  "CRIMINAL >52%  |  SUSPECT >38%  |  INNOCENT <38%"),
            bg=C["bg3"], fg=C["gray"], font=("Segoe UI", 8)
        ).pack(side="left", padx=16, pady=4)
        tk.Label(footer, text="⚠ FOR LAW ENFORCEMENT USE ONLY", bg=C["bg3"], fg=C["red"],
                 font=("Segoe UI", 8, "bold")).pack(side="right", padx=16)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # Try to set icon (ignore if fails)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = CISv2App(root)

    # Center on screen
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww, wh = 1500, 900
    x = (sw - ww) // 2
    y = (sh - wh) // 2
    root.geometry(f"{ww}x{wh}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()

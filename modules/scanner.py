"""
CIS v2 - Scanner Orchestrator
Manages the 3-scan pipeline, coordinates all analysis modules,
emits progress callbacks for the UI, and returns the final FusionResult.
"""
import cv2
import numpy as np
import threading
import time
import sys
import os

_CAM_LOCK = threading.Lock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import WEBCAM_SCAN_FRAMES, WEBCAM_TOTAL_SCANS, SCAN_INTERVAL_SEC, CROWD_THRESHOLD
from modules.face_analyzer import FaceAnalyzer, FaceFeatures
from modules.gait_analyzer import GaitAnalyzer, GaitFeatures
from modules.behavior_analyzer import BehaviorAnalyzer, BehaviorFeatures
from modules.fusion_engine import FusionEngine, FusionResult
from database.db_manager import log_detection_event


class ScanResult:
    """Full output of one complete scanning cycle."""
    def __init__(self):
        self.detections  : list  = []    # List of (FusionResult, FaceFeatures)
        self.scan_frames : list  = []    # (frame, label) tuples for display
        self.mode        : str   = ""
        self.input_source: str   = ""
        self.timestamp   : str   = ""
        self.success     : bool  = False
        self.error       : str   = ""


    @property
    def verdict(self) -> str:
        if not self.detections: return "CLEAR"
        # Return highest risk verdict found in crowd
        verdicts = [d[0].verdict for d in self.detections]
        if "CRIMINAL" in verdicts: return "CRIMINAL"
        if "WATCH LIST" in verdicts: return "WATCH LIST"
        return "CLEAR"

    @property
    def fusion_conf(self) -> float:
        if not self.detections: return 0.0
        return max(d[0].fusion_conf for d in self.detections)

    @property
    def is_crowd(self) -> bool:
        """True if the number of detections meets the crowd threshold."""
        return len(self.detections) >= CROWD_THRESHOLD



# --- Helper for Satellite Cameras ---
class SatelliteCap:
    """Mock VideoCapture for satellite streams."""
    def __init__(self, recv, cid):
        self.recv = recv
        self.cid = cid
        self.last_f = None
    def isOpened(self): 
        return self.recv is not None
    def read(self):
        if not self.recv: return False, None
        f = self.recv.get_frame(self.cid)
        if f is None or f is self.last_f: return False, None
        self.last_f = f
        return True, f.copy()
    def release(self): 
        pass


class Scanner:
    """
    Orchestrates the multi-scan identification pipeline.

    on_progress callback signature:
        callback(scan_num: int, total_scans: int, label: str, frame: np.ndarray)
    on_complete callback signature:
        callback(scan_result: ScanResult)
    """

    def __init__(self, on_progress=None, on_complete=None, on_frame=None):
        self.on_progress = on_progress   # Progress callback
        self.on_complete = on_complete   # Final result callback
        self.on_frame    = on_frame      # Per-frame display callback

        self.face_analyzer     = FaceAnalyzer()
        self.gait_analyzer     = GaitAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fusion_engine     = FusionEngine()

        self._stop_flag = threading.Event()

    def stop(self):
        """Signal the scanner to stop."""
        self._stop_flag.set()

    def reset(self):
        """Reset all modules for a fresh scan."""
        self._stop_flag.clear()
        self.gait_analyzer.reset()
        self.behavior_analyzer.reset()
        self.fusion_engine.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────────

    def scan_image(self, image_path: str) -> ScanResult:
        """Analyze a single image file."""
        from datetime import datetime
        self.reset()
        sr = ScanResult()
        sr.mode         = "image"
        sr.input_source = image_path
        sr.timestamp    = datetime.now().isoformat()

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                sr.error = f"Could not read image: {image_path}"
                return sr

            self._emit_progress(5, 100, "Scanning image pixels...", frame)
            faces = self.face_analyzer.analyze(frame)
            self._emit_progress(15, 100, f"Face detection complete: {len(faces)} found", frame)
            sr.detections = []
            
            num_faces = len(faces)
            for i, ff in enumerate(faces):
                current = i + 1
                # Map 1-N faces into the 15% to 75% range
                prog_val = 15 + int((current / max(num_faces, 1)) * 60)
                self._emit_progress(prog_val, 100, f"Analyzing Face {current}/{num_faces}...", frame)
                
                gf = self.gait_analyzer.analyze_frame(frame)
                bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)

                # 1.0 = Image Source (Trigger Famous Criminals in Web API)
                if len(ff.feature_vector) >= 27:
                    ff.feature_vector[24] = 1.0 
                    ff.feature_vector[25] = gf.confidence
                    ff.feature_vector[26] = bf.confidence

                fr = self.fusion_engine.fuse(
                    face_conf      = ff.confidence,
                    gait_conf      = gf.confidence,
                    behavior_conf  = bf.confidence,
                    face_features  = ff.feature_vector,
                    face_available = ff.detected,
                    gait_available = gf.detected,
                    behavior_available = bf.detected,
                    scan_num       = 1,
                )
                ff.verdict = fr.verdict # Attach for overlay
                sr.detections.append((fr, ff))

            sr.success  = True

            # Log detections
            self._log(sr)
            self._emit_complete(sr)


        except Exception as e:
            sr.error = str(e)
            import traceback
            traceback.print_exc()

        return sr

    def scan_video(self, video_path, on_frame_result=None, remote_receiver=None) -> ScanResult:
        """Analyze a video file, local webcam (int), or satellite stream using a threaded pipeline."""
        from datetime import datetime
        import queue
        from config.settings import VIDEO_SAMPLE_EVERY_N, FPS_TARGET
        
        self.reset()
        sr = ScanResult()
        sr.mode         = "video"
        sr.input_source = str(video_path)
        sr.timestamp    = datetime.now().isoformat()

        try:
            is_stream = False
            if isinstance(video_path, str) and video_path.startswith("SATELLITE:"):
                remote_id = int(video_path.split(":")[1])
                is_stream = True
                cap = SatelliteCap(remote_receiver, remote_id)
            else:
                cap = cv2.VideoCapture(video_path)
                is_stream = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0 or isinstance(video_path, int)

            if not cap.isOpened():
                sr.error = f"Cannot open source: {video_path}"
                return sr

            # Thread-safe communication
            frame_queue = queue.Queue(maxsize=2)
            results_queue = queue.Queue(maxsize=1)
            stop_signal = threading.Event()
            
            # --- Thread 1: Capture ---
            def capture_worker():
                f_idx = 0
                while not stop_signal.is_set() and not self._stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        if is_stream: time.sleep(0.01); continue
                        break
                    f_idx += 1
                    if frame_queue.full():
                        try: frame_queue.get_nowait()
                        except: pass
                    frame_queue.put((f_idx, frame))
                stop_signal.set()

            # --- Thread 2: Analysis ---
            def analysis_worker():
                last_analyzed_idx = -1
                while not stop_signal.is_set() and not self._stop_flag.is_set():
                    try:
                        f_idx, frame = frame_queue.get(timeout=0.2)
                    except queue.Empty: continue

                    # Analyze every Nth frame
                    if f_idx % VIDEO_SAMPLE_EVERY_N == 0:
                        faces = self.face_analyzer.analyze(frame)
                        current_detections = []
                        for ff in faces:
                            gf = self.gait_analyzer.analyze_frame(frame)
                            bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)
                            fr = self.fusion_engine.fuse(
                                face_conf=ff.confidence, gait_conf=gf.confidence, behavior_conf=bf.confidence,
                                face_features=ff.feature_vector if ff.detected else None,
                                face_available=ff.detected, scan_num=f_idx,
                                eye_count=ff.eye_count, face_ratio=ff.face_ratio
                            )
                            ff.verdict = fr.verdict
                            current_detections.append((fr, ff))
                        
                        if results_queue.full():
                            try: results_queue.get_nowait()
                            except: pass
                        results_queue.put(current_detections)

            # Start workers
            t_cap = threading.Thread(target=capture_worker, daemon=True)
            t_ana = threading.Thread(target=analysis_worker, daemon=True)
            t_cap.start(); t_ana.start()

            # --- Main Loop (Display & Callback) ---
            fr_last = FusionResult()
            detections_last = []
            frame_delay = 1.0 / FPS_TARGET

            while not stop_signal.is_set() or not frame_queue.empty():
                if self._stop_flag.is_set(): break
                
                try:
                    f_idx, frame = frame_queue.get(timeout=0.03)
                except queue.Empty: continue

                # Check for new results
                if not results_queue.empty():
                    detections_last = results_queue.get()
                    if detections_last:
                        fr_last = max(detections_last, key=lambda x: x[0].fusion_conf)[0]
                        if on_frame_result: on_frame_result(fr_last)

                # Overlay & Hud
                annotated = self.face_analyzer.draw_overlays(frame, [d[1] for d in detections_last])
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
                self._add_hud(annotated, fr_last, f_idx, total_f)
                
                if self.on_frame:
                    self.on_frame(annotated, detections_last)
                
                # Small wait to maintain target FPS
                time.sleep(frame_delay * 0.5)

            stop_signal.set()
            cap.release()
            sr.detections = detections_last
            sr.success = True
            self._log(sr)
            self._emit_complete(sr)

        except Exception as e:
            sr.error = str(e)
            import traceback; traceback.print_exc()

        return sr

    @staticmethod
    def _open_camera(preferred_index: int = 0):
        """
        Try to open the first working camera with the best backend for this OS.
        - On Windows: tries DirectShow (CAP_DSHOW) first for laptop webcams, then ANY.
        - preferred_index=-1  → auto-scan all indices 0..4.
        - Uses MJPEG codec hint for faster laptop camera startup on Windows.
        - Retries frame read up to 10 times for slow-starting cameras.
        Returns (cap, actual_index) or (None, -1).
        """
        import platform
        is_windows = platform.system() == "Windows"

        if preferred_index < 0:
            indices_to_try = list(range(5))
        else:
            indices_to_try = [preferred_index] + [i for i in range(5) if i != preferred_index]

        backends = ([cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if is_windows else [cv2.CAP_ANY])

        for idx in indices_to_try:
            for backend in backends:
                try:
                    with _CAM_LOCK:
                        cap = cv2.VideoCapture(idx, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue

                    # Set resolution and framerate
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    # Prefer MJPEG for faster laptop-cam startup
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                    # Retry up to 10 frames — some cameras take time to warm up
                    for _ in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            bname = 'DSHOW' if backend == cv2.CAP_DSHOW else 'ANY'
                            print(f"[Camera] Opened index={idx} backend={bname} "
                                  f"res={frame.shape[1]}x{frame.shape[0]}")
                            return cap, idx
                    cap.release()
                except Exception as exc:
                    print(f"[Camera] Error trying index={idx}: {exc}")
        return None, -1

    def scan_webcam(self, camera_index: int = 0, remote_id: int = 1, remote_receiver = None) -> ScanResult:
        """
        Perform the 3-scan webcam cycle with threaded capture to eliminate lag.
        Supports local camera (remote_id=1) or Satellite cameras (remote_id=2,3).
        """
        from datetime import datetime
        import queue
        self.reset()
        sr = ScanResult()
        sr.mode         = "webcam"
        sr.input_source = f"CAM:{camera_index}" if remote_id == 1 else f"REMOTE:{remote_id}"
        sr.timestamp    = datetime.now().isoformat()

        cap = None
        stop_capture = threading.Event()
        latest_frame_q = queue.Queue(maxsize=1)

        try:
            if remote_id == 1:
                cap, actual_index = self._open_camera(camera_index)
                if cap is None:
                    sr.error = ("No camera found. Please ensure your webcam is connected.")
                    self._emit_complete(sr)
                    return sr
                sr.input_source = f"CAM:{actual_index}"
            else:
                cap = SatelliteCap(remote_receiver, remote_id)

            def capture_worker():
                while not stop_capture.is_set() and not self._stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    
                    if latest_frame_q.full():
                        try: latest_frame_q.get_nowait()
                        except: pass
                    latest_frame_q.put(frame)
                    time.sleep(0.01)

            # Start capture thread
            t_cap = threading.Thread(target=capture_worker, daemon=True)
            t_cap.start()

            # Warm-up/Initial display
            start_wait = time.time()
            while time.time() - start_wait < 1.0:
                try:
                    warm_frame = latest_frame_q.get(timeout=0.1)
                    if self.on_frame: self.on_frame(warm_frame, None)
                except queue.Empty: pass

            # 3 scan passes
            for scan_num in range(1, WEBCAM_TOTAL_SCANS + 1):
                if self._stop_flag.is_set(): break

                scan_label = f"SCAN {scan_num}/{WEBCAM_TOTAL_SCANS}"
                self._emit_progress(scan_num, WEBCAM_TOTAL_SCANS, scan_label, None)

                ff_best = None
                gf_last = GaitFeatures()
                bf_last = BehaviorFeatures()
                fr_last = None

                for f_idx in range(WEBCAM_SCAN_FRAMES):
                    if self._stop_flag.is_set(): break

                    try:
                        frame = latest_frame_q.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    # Analyze
                    faces = self.face_analyzer.analyze(frame)
                    ff = faces[0] if faces else FaceFeatures()
                    gf = self.gait_analyzer.analyze_frame(frame)
                    bf = self.behavior_analyzer.analyze_frame(frame, face_bbox=ff.bbox)

                    # Keep best-quality face
                    if ff.detected and (ff_best is None or ff.quality > ff_best.quality):
                        ff_best = ff

                    gf_last = gf
                    bf_last = bf

                    # Annotate frame for display
                    annotated = self.face_analyzer.draw_overlays(frame, faces, scan_label=scan_label)
                    self._add_scan_hud(annotated, scan_num, f_idx, WEBCAM_SCAN_FRAMES)
                    if self.on_frame:
                        self.on_frame(annotated, None)

                # Run fusion for this scan pass
                ff_use = ff_best if ff_best else FaceFeatures()
                if ff_use.detected and len(ff_use.feature_vector) >= 27:
                    ff_use.feature_vector[24] = 0.0 # Webcam source
                    ff_use.feature_vector[25] = gf_last.confidence
                    ff_use.feature_vector[26] = bf_last.confidence

                fr_last = self.fusion_engine.fuse(
                    face_conf      = ff_use.confidence,
                    gait_conf      = gf_last.confidence,
                    behavior_conf  = bf_last.confidence,
                    face_features  = ff_use.feature_vector if ff_use.detected else None,
                    behavior_available = bf_last.detected,
                    scan_num       = scan_num,
                    eye_count      = ff_use.eye_count,
                    face_ratio     = ff_use.face_ratio,
                )

                # Brief pause between scans
                if scan_num < WEBCAM_TOTAL_SCANS:
                    pause_end = time.time() + SCAN_INTERVAL_SEC
                    while time.time() < pause_end and not self._stop_flag.is_set():
                        try:
                            frame = latest_frame_q.get(timeout=0.05)
                            # Show "PAUSE" screen between scans
                            _f = frame.copy()
                            cv2.rectangle(_f, (0, 0), (_f.shape[1], 60), (30, 30, 30), -1)
                            cv2.putText(_f, f"Scan {scan_num} complete. Preparing scan {scan_num+1}...",
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                            if self.on_frame:
                                self.on_frame(_f, fr_last)
                        except queue.Empty:
                            pass

            stop_capture.set()
            if cap: cap.release()

            # Final aggregated result
            final_fr = self.fusion_engine.aggregate_scans()
            sr.fusion   = final_fr
            sr.face     = ff_use if 'ff_use' in dir() else FaceFeatures()
            sr.gait     = gf_last if 'gf_last' in dir() else GaitFeatures()
            sr.behavior = bf_last if 'bf_last' in dir() else BehaviorFeatures()
            sr.success  = True
            
            # Attach the aggregated result to sr.detections for UI consistency
            sr.detections = [(final_fr, sr.face)]
            
            self._log(sr)
            self._emit_complete(sr)

        except Exception as e:
            stop_capture.set()
            if cap: cap.release()
            sr.error = str(e)
            import traceback
            traceback.print_exc()

        return sr

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _emit_progress(self, current, total, label, frame):
        if self.on_progress:
            try:
                self.on_progress(current, total, label, frame)
            except Exception:
                pass

    def _emit_complete(self, sr: ScanResult):
        if self.on_complete:
            try:
                self.on_complete(sr)
            except Exception:
                pass

    def _log(self, sr: ScanResult):
        """Log this scan to the database."""
        try:
            fr = sr.fusion
            event = {
                "criminal_id":  fr.suspect_id,
                "mode":         sr.mode,
                "face_conf":    fr.face_conf,
                "gait_conf":    fr.gait_conf,
                "behavior_conf":fr.behavior_conf,
                "fusion_conf":  fr.fusion_conf,
                "verdict":      fr.verdict,
                "risk_level":   fr.risk_level,
                "input_source": sr.input_source,
                "notes":        f"Match: {fr.suspect_name or 'None'} | DB sim: {fr.db_similarity:.2%}",
            }
            log_detection_event(event)
        except Exception:
            pass

    def _add_hud(self, frame: np.ndarray, fr: FusionResult, frame_num: int, total: int):
        """Add HUD overlay for video mode."""
        h, w = frame.shape[:2]
        # Progress bar
        prog = frame_num / max(total, 1)
        bar_w = int(w * prog)
        cv2.rectangle(frame, (0, h - 6), (w, h), (30, 30, 30), -1)
        color = (0, 0, 255) if fr.verdict == "CRIMINAL" else (0, 165, 255) if fr.verdict == "WATCH LIST" else (0, 200, 60)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), color, -1)
        # Top label
        cv2.rectangle(frame, (0, 0), (380, 38), (0, 0, 0), -1)
        name = fr.suspect_name if fr.suspect_name else fr.verdict
        sim_str = f" ({fr.db_similarity:.0%})" if fr.db_similarity > 0 else ""
        label = f"{name}{sim_str}  {fr.fusion_conf:.0%}"
        cv2.putText(frame, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)

    def _add_scan_hud(self, frame: np.ndarray, scan_num: int, frame_idx: int, total_frames: int):
        """Add HUD overlay for webcam scan mode."""
        h, w = frame.shape[:2]
        progress = frame_idx / max(total_frames, 1)
        bar_w = int(w * progress)
        # Scanning progress bar (orange)
        cv2.rectangle(frame, (0, h - 8), (w, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 165, 255), -1)
        # Scan counter top-right
        cv2.rectangle(frame, (w - 180, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"SCAN {scan_num}/{WEBCAM_TOTAL_SCANS}", (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

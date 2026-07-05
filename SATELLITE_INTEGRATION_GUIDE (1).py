# CIS v2 - SATELLITE INTEGRATION MODULE
# Copy these blocks into your existing cis_v2.py to enable Multi-Cam & Remote Registration

import cv2
import numpy as np
import socket
import struct
import pickle
import threading
from database.db_manager import add_criminal_to_db # Ensure this exists

# --- BLOCK 1: UPGRADED STREAM RECEIVER ---
# Replace your old StreamReceiver class with this one
class StreamReceiver:
    def __init__(self, app, host='0.0.0.0', port=9999):
        self.app = app
        self.host = host
        self.port = port
        self.frames = {1: None, 2: None, 3: None}
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        while self._running:
            conn, addr = server_socket.accept()
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn):
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(1024)
                if not chunk: break
                data += chunk
            
            # REMOTE REGISTRATION HANDLER
            if b"POST /register" in data:
                headers, body = data.split(b"\r\n\r\n", 1)
                name = "Unknown"
                for line in headers.decode().split("\r\n"):
                    if "Name:" in line: name = line.split(":")[1].strip()
                
                frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # AI Processing on main PC
                    from modules.scanner import FaceFeatures
                    ff = FaceFeatures()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ff.feature_vector = self.app.face_analyzer._build_feature_vector(ff, gray)
                    if ff.feature_vector:
                        add_criminal_to_db(name, face_features=ff.feature_vector)
                        self.app.activity_log.log(f"Remote Register: {name}", "info")
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

            # FRAME UPLOAD HANDLER
            elif b"POST /upload" in data:
                headers, body = data.split(b"\r\n\r\n", 1)
                cam_id = 2 if b"cam=2" in headers else 3
                frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None: self.frames[cam_id] = frame
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        except: pass
        finally: conn.close()

    def get_frame(self, cam_id):
        return self.frames.get(cam_id)

# --- BLOCK 2: WEBCAM TAB SWITCHING LOGIC ---
# Add this to your WebcamTab's _build method to enable camera switching
# self._active_cam_id = tk.IntVar(value=1)
# cam_row = tk.Frame(parent)
# tk.Radiobutton(cam_row, text="Local", variable=self._active_cam_id, value=1).pack(side="left")
# tk.Radiobutton(cam_row, text="Remote 2", variable=self._active_cam_id, value=2).pack(side="left")
# tk.Radiobutton(cam_row, text="Remote 3", variable=self._active_cam_id, value=3).pack(side="left")

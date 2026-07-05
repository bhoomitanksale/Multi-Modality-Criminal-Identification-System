import cv2
import requests
import time
import socket
import threading
import tkinter as tk
import queue

# --- CONFIGURATION ---
CAM_ID = "2"  # Choose '2' or '3' for the camera slot

def discover_hub_once(timeout=2.0):
    """Attempts to find the Hub once, returning IP, Port. Returns None, None if not found."""
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_sock.settimeout(timeout)
    try:
        udp_sock.sendto(b"DISCOVER_CIS_HUB", ('255.255.255.255', 8082))
        udp_sock.sendto(b"DISCOVER_CIS_HUB", ('<broadcast>', 8082))
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            subnet_broadcast = ".".join(local_ip.split(".")[:3]) + ".255"
            udp_sock.sendto(b"DISCOVER_CIS_HUB", (subnet_broadcast, 8082))
        except: pass

        data, addr = udp_sock.recvfrom(1024)
        data_str = data.decode('utf-8')
        if data_str.startswith("CIS_HUB:"):
            parts = data_str.split(":")
            return parts[1], parts[2]
    except:
        pass
    return None, None

class SatelliteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CIS Satellite Sender")
        self.root.geometry("340x310")
        self.root.configure(bg="#0d0d1a")
        self.root.resizable(False, False)
        
        self.is_running = False
        self.stream_thread = None
        
        # UI Elements
        tk.Label(root, text="🛰️ SATELLITE DEVICE", fg="#00d4ff", bg="#0d0d1a", font=("Segoe UI", 16, "bold")).pack(pady=(20, 5))
        
        self.lbl_status = tk.Label(root, text="STATUS: IDLE", fg="#7a7a9a", bg="#0d0d1a", font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(pady=(0, 10))
        
        # Camera ID Selection
        self.cam_id_var = tk.StringVar(value="2")
        self.frame_r = tk.Frame(root, bg="#0d0d1a")
        self.frame_r.pack(pady=(0, 15))
        
        self.rb2 = tk.Radiobutton(self.frame_r, text="Remote 2", variable=self.cam_id_var, value="2",
                       bg="#0d0d1a", fg="#f0f0f0", selectcolor="#252545", activebackground="#0d0d1a",
                       activeforeground="#00d4ff", font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.rb2.pack(side="left", padx=10)
        self.rb3 = tk.Radiobutton(self.frame_r, text="Remote 3", variable=self.cam_id_var, value="3",
                       bg="#0d0d1a", fg="#f0f0f0", selectcolor="#252545", activebackground="#0d0d1a",
                       activeforeground="#00d4ff", font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.rb3.pack(side="left", padx=10)
        
        # Buttons
        self.btn_start = tk.Button(root, text="▶ START STREAM", command=self.start_stream, 
                                   bg="#00e676", fg="#0d0d1a", font=("Segoe UI", 12, "bold"),
                                   relief="flat", activebackground="#00c853", cursor="hand2")
        self.btn_start.pack(pady=6, fill="x", padx=50, ipady=5)
        
        self.btn_stop = tk.Button(root, text="⏹ STOP STREAM", command=self.stop_stream, state=tk.DISABLED,
                                  bg="#252545", fg="#7a7a9a", font=("Segoe UI", 12, "bold"),
                                  relief="flat", activebackground="#ff3355", cursor="hand2")
        self.btn_stop.pack(pady=6, fill="x", padx=50, ipady=5)

    def update_status(self, text, color):
        self.lbl_status.config(text=f"STATUS: {text}", fg=color)

    def start_stream(self):
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED, bg="#252545", fg="#7a7a9a")
        self.btn_stop.config(state=tk.NORMAL, bg="#ff3355", fg="white")
        self.rb2.config(state=tk.DISABLED)
        self.rb3.config(state=tk.DISABLED)
        self.update_status("CONNECTING...", "#ff9900")
        self.stream_thread = threading.Thread(target=self.stream_worker, daemon=True)
        self.stream_thread.start()

    def stop_stream(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL, bg="#00e676", fg="#0d0d1a")
        self.btn_stop.config(state=tk.DISABLED, bg="#252545", fg="#7a7a9a")
        self.rb2.config(state=tk.NORMAL)
        self.rb3.config(state=tk.NORMAL)
        self.update_status("IDLE", "#7a7a9a")

    def stream_worker(self):
        hub_ip, hub_port = None, None
        # 1. Discovery Loop
        while self.is_running and hub_ip is None:
            self.root.after(0, lambda: self.update_status("SEARCHING FOR HUB...", "#ff9900"))
            hub_ip, hub_port = discover_hub_once()
            if not hub_ip:
                time.sleep(0.5)

        if not self.is_running:
            return

        # 2. Capture Loop
        cam_id = self.cam_id_var.get()
        url = f"http://{hub_ip}:{hub_port}/upload?cam={cam_id}"
        self.root.after(0, lambda: self.update_status(f"STREAMING TO {hub_ip}", "#00d4ff"))
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.root.after(0, lambda: self.update_status("ERROR: WEBCAM OFFLINE", "#ff3355"))
            self.root.after(2000, self.stop_stream)
            return

        try:
            # Use a queue for frames to send, max size 1 to always send latest
            send_queue = queue.Queue(maxsize=1)

            def sender_thread():
                while self.is_running:
                    try:
                        f = send_queue.get(timeout=0.1)
                        _, img_encoded = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        requests.post(url, data=img_encoded.tobytes(), timeout=0.1)
                    except queue.Empty:
                        continue
                    except Exception:
                        pass

            threading.Thread(target=sender_thread, daemon=True).start()

            while self.is_running:
                ret, frame = cap.read()
                if not ret: 
                    time.sleep(0.01)
                    continue

                frame = cv2.resize(frame, (640, 480))
                # Update UI with camera feed if needed (optional, for local feedback)
                
                # Push to sender queue (don't block)
                if send_queue.full():
                    try: send_queue.get_nowait()
                    except: pass
                send_queue.put(frame)

                # Slow down slightly to ~25 FPS but don't hard sleep 0.05
                time.sleep(0.03)
        finally:
            cap.release()
            self.root.after(0, self.stop_stream)

if __name__ == "__main__":
    import queue
    root = tk.Tk()
    app = SatelliteApp(root)
    root.mainloop()

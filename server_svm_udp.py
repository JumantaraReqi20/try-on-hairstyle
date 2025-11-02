import cv2
import socket
import struct
import threading
import time
import math
import numpy as np
import joblib
import logging
from pathlib import Path

# --- Konfigurasi Awal (Gabungan dari kedua proyek) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Konfigurasi Jaringan (dari webcam_server_udp.py)
HOST = '127.0.0.1'  # localhost
PORT = 8888
MAX_PACKET_SIZE = 60000  # ~60KB per paket

# Konfigurasi Model CV (dari run_webcam.py)
MODEL_DIR = Path("models")
ASSETS_DIR = Path("assets")
K_CLUSTERS = 512
IMG_SIZE = (128, 128)
ORB_FEATURES = 500

# --- Fungsi Helper CV (dari run_webcam.py) ---
# (Kita letakkan di luar class agar bersih)

def create_orb_detector(n_features=ORB_FEATURES):
    return cv2.ORB_create(nfeatures=n_features)

def extract_descriptors(img_gray, orb, size=IMG_SIZE):
    try:
        img = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
        _, descriptors = orb.detectAndCompute(img, None)
        return descriptors
    except Exception:
        return None

def compute_bovw_histogram(descriptors, codebook):
    k = codebook.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k, dtype=np.float32)
    indices = codebook.predict(descriptors)
    hist, _ = np.histogram(indices, bins=np.arange(k + 1))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist, ord=1)
    if norm > 0:
        hist /= norm
    return hist

def overlay_hairstyle(frame, hairstyle_img, face_box, y_offset_factor=0.6, scale_factor=1.5):
    # (Fungsi ini sama persis seperti di run_webcam.py)
    x, y, w, h = face_box
    hair_h_orig, hair_w_orig, _ = hairstyle_img.shape
    target_hair_w = int(w * scale_factor)
    target_hair_h = int(hair_h_orig * (target_hair_w / hair_w_orig))
    if target_hair_w == 0 or target_hair_h == 0:
        return frame
    hair_resized = cv2.resize(hairstyle_img, (target_hair_w, target_hair_h))
    pos_x = x - (target_hair_w - w) // 2
    pos_y = y - int(target_hair_h * y_offset_factor)
    frame_h, frame_w, _ = frame.shape
    x1_frame = max(pos_x, 0)
    y1_frame = max(pos_y, 0)
    x2_frame = min(pos_x + target_hair_w, frame_w)
    y2_frame = min(pos_y + target_hair_h, frame_h)
    x1_hair = max(0, -pos_x)
    y1_hair = max(0, -pos_y)
    x2_hair = x1_hair + (x2_frame - x1_frame)
    y2_hair = y1_hair + (y2_frame - y1_frame)
    if (x2_frame <= x1_frame) or (y2_frame <= y1_frame):
        return frame
    frame_roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    hair_roi = hair_resized[y1_hair:y2_hair, x1_hair:x2_hair]
    alpha_mask = hair_roi[:, :, 3] / 255.0
    rgb_hair = hair_roi[:, :, :3]
    alpha_3d = np.stack([alpha_mask] * 3, axis=-1)
    frame_roi_blended = (rgb_hair * alpha_3d) + (frame_roi * (1.0 - alpha_3d))
    frame[y1_frame:y2_frame, x1_frame:x2_frame] = frame_roi_blended.astype(np.uint8)
    return frame

# --- Class Server Utama (Gabungan) ---
class SVM_WebcamServerUDP:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = set()  # Set untuk menyimpan alamat client (dari webcam_server_udp.py)
        self.cap = None
        self.running = False
        self.sequence_number = 0
        self.max_packet_size = MAX_PACKET_SIZE
        
        # Variabel CV
        self.orb = None
        self.haar_cascade = None
        self.codebook = None
        self.scaler = None
        self.svm = None
        self.hairstyle_list = []
        self.current_hairstyle_index = 0
        
        # Variabel Smoothing (dari run_webcam.py)
        self.last_known_box = None
        self.frames_missed = 0
        self.MAX_FRAMES_MISS = 10

    def load_cv_models(self):
        """Memuat semua model CV dan aset hairstyle."""
        logging.info("Memuat model CV...")
        try:
            self.orb = create_orb_detector()
            self.haar_cascade = cv2.CascadeClassifier(str(ASSETS_DIR / "haarcascade_frontalface_default.xml"))
            self.codebook = joblib.load(MODEL_DIR / "codebook.pkl")
            self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            self.svm = joblib.load(MODEL_DIR / "svm.pkl")
            
            # Load hairstyles (dari run_webcam.py)
            hairstyle_paths = sorted(list(ASSETS_DIR.glob("hairstyle_*.png")))
            if not hairstyle_paths:
                raise IOError("Tidak ada file hairstyle (hairstyle_*.png) ditemukan.")
                
            for path in hairstyle_paths:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:
                    self.hairstyle_list.append(img)
            
            if not self.hairstyle_list:
                raise IOError("Gagal memuat gambar hairstyle yang valid.")
                
            logging.info(f"Berhasil memuat {len(self.hairstyle_list)} hairstyle dan model CV.")
            return True
            
        except Exception as e:
            logging.error(f"Gagal memuat model CV: {e}")
            return False

    def start_server(self):
        """Memulai server UDP dan webcam."""
        if not self.load_cv_models():
            return
            
        try:
            # Buat UDP socket (dari webcam_server_udp.py)
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            logging.info(f"🚀 Server SVM-UDP dimulai di {self.host}:{self.port}")
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("❌ Error: Tidak dapat mengakses webcam")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            
            # Thread untuk menerima pesan (REGISTER, NEXT_HAIR, dll)
            listen_thread = threading.Thread(target=self.listen_for_clients)
            listen_thread.start()
            
            # Thread untuk memproses CV dan mengirim frame
            stream_thread = threading.Thread(target=self.stream_webcam_cv)
            stream_thread.start()
            
        except Exception as e:
            logging.error(f"❌ Error starting server: {e}")
    
    def listen_for_clients(self):
        """Mendengarkan pesan dari client."""
        self.server_socket.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                message = data.decode('utf-8')
                
                if message == "REGISTER": #
                    if addr not in self.clients:
                        self.clients.add(addr)
                        logging.info(f"✅ Client terdaftar: {addr} | Total: {len(self.clients)}")
                        self.server_socket.sendto("REGISTERED".encode('utf-8'), addr)
                
                elif message == "UNREGISTER": #
                    if addr in self.clients:
                        self.clients.remove(addr)
                        logging.info(f"🔌 Client keluar: {addr} | Total: {len(self.clients)}")
                
                # --- INI ADALAH MODIFIKASI KITA ---
                elif message == "NEXT_HAIR":
                    if addr in self.clients: # Hanya client terdaftar yang bisa ganti
                        self.current_hairstyle_index = (self.current_hairstyle_index + 1) % len(self.hairstyle_list)
                        logging.info(f"Client {addr} ganti ke hairstyle {self.current_hairstyle_index + 1}")
                
                elif message == "PREV_HAIR":
                    if addr in self.clients:
                        self.current_hairstyle_index = (self.current_hairstyle_index - 1) % len(self.hairstyle_list)
                        logging.info(f"Client {addr} ganti ke hairstyle {self.current_hairstyle_index + 1}")
                # --- BATAS MODIFIKASI ---

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.warning(f"⚠️ Error di listen_thread: {e}")

    def stream_webcam_cv(self):
        """
        Ini adalah gabungan dari run_webcam.py (logika CV) 
        dan stream_webcam dari webcam_server_udp.py (logika encoding).
        """
        while self.running:
            try:
                if len(self.clients) == 0:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("❌ Error: Gagal membaca frame webcam")
                    break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # --- Mulai Logika CV dari run_webcam.py ---
                kandidat_wajah = self.haar_cascade.detectMultiScale(
                    gray, 1.2, 4, minSize=(80, 80) # Pakai minNeighbors=4 agar lebih stabil
                )
                
                validated_faces = []
                for (x, y, w, h) in kandidat_wajah:
                    roi_gray = gray[y:y+h, x:x+w]
                    descriptors = extract_descriptors(roi_gray, self.orb, IMG_SIZE)
                    hist = compute_bovw_histogram(descriptors, self.codebook)
                    hist_scaled = self.scaler.transform(hist.reshape(1, -1))
                    if self.svm.predict(hist_scaled)[0] == 1:
                        validated_faces.append((x, y, w, h))

                current_box_to_draw = None
                if len(validated_faces) > 0:
                    best_face = max(validated_faces, key=lambda b: b[2] * b[3])
                    current_box_to_draw = best_face
                    self.last_known_box = best_face
                    self.frames_missed = 0
                elif self.last_known_box is not None and self.frames_missed < self.MAX_FRAMES_MISS:
                    current_box_to_draw = self.last_known_box
                    self.frames_missed += 1
                else:
                    self.last_known_box = None
                    self.frames_missed = 0
                
                # Ambil hairstyle yang aktif
                active_hairstyle = self.hairstyle_list[self.current_hairstyle_index]

                if current_box_to_draw is not None:
                    # Terapkan overlay ke frame
                    frame = overlay_hairstyle(
                        frame,
                        active_hairstyle,
                        current_box_to_draw,
                        y_offset_factor=0.45,
                        scale_factor=1
                    )
                # --- Selesai Logika CV ---
                
                # Encode frame (yang sudah di-overlay) ke JPEG
                # (dari webcam_server_udp.py)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # Kualitas 50% untuk kecepatan
                result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
                
                if result:
                    frame_data = encoded_img.tobytes()
                    # Kirim frame ter-encode via UDP
                    self.send_frame_to_clients(frame_data)
                
                time.sleep(0.033)  # Target ~30 FPS

            except Exception as e:
                logging.error(f"❌ Error di stream_thread: {e}")
                break

    def send_frame_to_clients(self, frame_data):
        """Mengirim frame data ke semua client dengan fragmentasi (dari webcam_server_udp.py)."""
        if not frame_data or len(self.clients) == 0:
            return
        
        self.sequence_number = (self.sequence_number + 1) % 65536
        frame_size = len(frame_data)
        
        header_size = 12  # 4b seq + 4b total + 4b index
        payload_size = self.max_packet_size - header_size
        total_packets = math.ceil(frame_size / payload_size)
        
        clients_to_remove = set()
        
        for client_addr in self.clients.copy():
            try:
                for packet_index in range(total_packets):
                    start_pos = packet_index * payload_size
                    end_pos = min(start_pos + payload_size, frame_size)
                    packet_data = frame_data[start_pos:end_pos]
                    
                    # Header: [sequence_number:4][total_packets:4][packet_index:4]
                    header = struct.pack("!III", self.sequence_number, total_packets, packet_index)
                    udp_packet = header + packet_data
                    
                    self.server_socket.sendto(udp_packet, client_addr)
            
            except Exception as e:
                logging.warning(f"Error mengirim ke {client_addr}: {e}")
                clients_to_remove.add(client_addr)
        
        for client_addr in clients_to_remove:
            if client_addr in self.clients:
                self.clients.remove(client_addr)

    def stop_server(self):
        """Menghentikan server."""
        logging.info("⏹️  Menghentikan server...")
        self.running = False
        
        for client_addr in self.clients.copy():
            self.server_socket.sendto("SERVER_SHUTDOWN".encode('utf-8'), client_addr)
        
        self.clients.clear()
        if self.server_socket:
            self.server_socket.close()
        if self.cap:
            self.cap.release()
        logging.info("✅ Server dihentikan.")

# --- Titik Masuk Program ---
if __name__ == "__main__":
    server = SVM_WebcamServerUDP()
    try:
        server.start_server()
        logging.info("📺 Server berjalan. Tekan Ctrl+C untuk berhenti.")
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop_server()
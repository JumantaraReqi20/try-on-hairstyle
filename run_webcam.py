import cv2
import numpy as np
import joblib
import logging
import time
from pathlib import Path

# --- Konfigurasi Awal ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MODEL_DIR = Path("models")
ASSETS_DIR = Path("assets")

# Pastikan semua file ada
HAAR_CASCADE_PATH = ASSETS_DIR / "haarcascade_frontalface_default.xml"
CODEBOOK_PATH = MODEL_DIR / "codebook.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
SVM_PATH = MODEL_DIR / "svm.pkl"

# Cek file
if not all([p.exists() for p in [HAAR_CASCADE_PATH, CODEBOOK_PATH, SCALER_PATH, SVM_PATH]]):
    logging.error("File model atau aset tidak ditemukan! Pastikan Anda sudah menjalankan train.py dan file ada di folder assets/.")
    exit()

# Konfigurasi Model (HARUS SAMA DENGAN SAAT TRAINING)
K_CLUSTERS = 512
IMG_SIZE = (128, 128)
ORB_FEATURES = 500

# --- 1. Fungsi Helper (Sama seperti training.py) ---

def create_orb_detector(n_features=ORB_FEATURES):
    """Membuat detektor ORB."""
    return cv2.ORB_create(nfeatures=n_features)

def extract_descriptors(img_gray, orb, size=IMG_SIZE):
    """Mengekstrak 'sidik jari' dari gambar (bukan path)."""
    try:
        img = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return descriptors
    except Exception:
        return None

def compute_bovw_histogram(descriptors, codebook):
    """Menghitung 1 histogram (vektor) untuk 1 gambar."""
    k = codebook.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k, dtype=np.float32) # Vektor kosong

    indices = codebook.predict(descriptors)
    hist, _ = np.histogram(indices, bins=np.arange(k + 1))
    
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist, ord=1)
    if norm > 0:
        hist /= norm
        
    return hist

# --- 2. Fungsi Overlay (Menempel Hairstyle) ---

def overlay_hairstyle(frame, hairstyle_img, face_box, y_offset_factor=0.6, scale_factor=1.5):
    """Menempelkan gambar hairstyle (PNG) ke frame."""
    x, y, w, h = face_box
    
    # --- 1. Skala Hairstyle ---
    hair_h_orig, hair_w_orig, _ = hairstyle_img.shape
    
    target_hair_w = int(w * scale_factor)
    target_hair_h = int(hair_h_orig * (target_hair_w / hair_w_orig))
    
    if target_hair_w == 0 or target_hair_h == 0:
        return frame
        
    hair_resized = cv2.resize(hairstyle_img, (target_hair_w, target_hair_h))

    # --- 2. Posisi Hairstyle ---
    # Atur posisi y berdasarkan offset dari atas kotak wajah
    # y_offset_factor = 0.6 -> 60% tinggi rambut akan berada DI ATAS garis y
    pos_x = x - (target_hair_w - w) // 2
    pos_y = y - int(target_hair_h * y_offset_factor)
    
    # --- 3. Alpha Blending (Menempel PNG transparan) ---
    frame_h, frame_w, _ = frame.shape
    
    # Tentukan area di frame
    x1_frame = max(pos_x, 0)
    y1_frame = max(pos_y, 0)
    x2_frame = min(pos_x + target_hair_w, frame_w)
    y2_frame = min(pos_y + target_hair_h, frame_h)
    
    # Tentukan area di rambut (jika terpotong)
    x1_hair = max(0, -pos_x)
    y1_hair = max(0, -pos_y)
    x2_hair = x1_hair + (x2_frame - x1_frame)
    y2_hair = y1_hair + (y2_frame - y1_frame)
    
    if (x2_frame <= x1_frame) or (y2_frame <= y1_frame):
        return frame

    # Ambil ROI (Region of Interest)
    frame_roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    hair_roi = hair_resized[y1_hair:y2_hair, x1_hair:x2_hair]
    
    # Pisahkan Alpha (transparansi)
    alpha_mask = hair_roi[:, :, 3] / 255.0  # (h, w)
    rgb_hair = hair_roi[:, :, :3]           # (h, w, 3)

    # Buat mask 3D
    alpha_3d = np.stack([alpha_mask] * 3, axis=-1)

    # Blend: (Rambut * Alpha) + (Frame * (1 - Alpha))
    frame_roi_blended = (rgb_hair * alpha_3d) + (frame_roi * (1.0 - alpha_3d))
    
    # Kembalikan ke frame
    frame[y1_frame:y2_frame, x1_frame:x2_frame] = frame_roi_blended.astype(np.uint8)
    
    return frame

# --- 3. Fungsi Utama Webcam ---

def main_webcam():
    logging.info("--- Memulai Aplikasi Webcam ---")
    
    # --- Langkah 3.1: Load Semua Model ---
    logging.info("Memuat semua model...")
    try:
        orb = create_orb_detector()
        haar_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        codebook = joblib.load(CODEBOOK_PATH)
        scaler = joblib.load(SCALER_PATH)
        svm = joblib.load(SVM_PATH)
        # --- Load SEMUA Hairstyle ---
        logging.info("Memuat aset hairstyle...")
        # Cari semua file .png yang diawali 'hairstyle_' dan urutkan
        hairstyle_paths = sorted(list(ASSETS_DIR.glob("hairstyle_*.png")))
        
        if not hairstyle_paths:
            logging.error("Tidak ada file hairstyle (contoh: hairstyle_1.png) ditemukan di folder assets/.")
            return

        hairstyle_list = [] # Daftar untuk menyimpan gambar
        for path in hairstyle_paths:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                hairstyle_list.append(img)
            else:
                logging.warning(f"Gagal memuat {path} atau bukan RGBA 4-channel. Dilewati.")
        
        if not hairstyle_list:
            logging.error("Gagal memuat gambar hairstyle yang valid.")
            return

        logging.info(f"Berhasil memuat {len(hairstyle_list)} hairstyle.")
        current_hairstyle_index = 0 # Mulai dari hairstyle pertama (index 0)
             
    except Exception as e:
        logging.error(f"Gagal memuat model atau aset: {e}")
        return
        
    logging.info("Model dan aset berhasil dimuat.")
    
    # --- Langkah 3.2: Buka Webcam ---
    cap = cv2.VideoCapture(0) # 0 = webcam default
    if not cap.isOpened():
        logging.error("Tidak bisa membuka webcam.")
        return

    logging.info("Webcam terbuka. Tekan 'q' untuk keluar.")
    
    # Variabel untuk FPS
    fps_start = time.time()
    fps_frames = 0
    fps_text = "FPS: 0"
    
    show_overlay = True # Tampilkan overlay (bisa di-toggle)

    last_known_box = None  # Untuk menyimpan kotak terakhir
    frames_missed = 0      # Penghitung frame gagal
    MAX_FRAMES_MISS = 10   # Maksimal "memori" (sekitar 1/3 detik)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Cerminkan frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Langkah 3.3: Deteksi Cepat (Haar) ---
        # Ini hanya "proposal" atau "kandidat"
        kandidat_wajah = haar_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=4, 
            minSize=(80, 80)
        )
        
        # --- Langkah 3.4 & 3.5: Validasi, Smoothing, dan Overlay ---
        
        # Coba cari wajah yang divalidasi SVM
        validated_faces = []
        for (x, y, w, h) in kandidat_wajah:
            roi_gray = gray[y:y+h, x:x+w]
            descriptors = extract_descriptors(roi_gray, orb)
            hist = compute_bovw_histogram(descriptors, codebook)
            hist_2d = hist.reshape(1, -1)
            hist_scaled = scaler.transform(hist_2d)
            
            if svm.predict(hist_scaled)[0] == 1:
                validated_faces.append((x, y, w, h))

        # --- Logika Smoothing (Memori) ---
        current_box_to_draw = None
        
        if len(validated_faces) > 0:
            # Sukses! Ambil wajah terbesar (jika ada banyak)
            best_face = max(validated_faces, key=lambda b: b[2] * b[3]) 
            current_box_to_draw = best_face
            last_known_box = best_face  # Perbarui "memori"
            frames_missed = 0           # Reset penghitung gagal
            
        elif last_known_box is not None and frames_missed < MAX_FRAMES_MISS:
            # GAGAL! Tapi kita masih punya "memori"
            # Gunakan kotak terakhir yang diketahui
            current_box_to_draw = last_known_box 
            frames_missed += 1 # Tambah hitungan gagal
            
        else:
            # GAGAL Total! Dan memori habis / tidak ada memori
            last_known_box = None # Hapus memori
            frames_missed = 0
        
        # --- Logika Overlay ---
        if current_box_to_draw is not None:
            # Gambar kotak (opsional)
            (x, y, w, h) = current_box_to_draw
            color = (0, 255, 0) if frames_missed == 0 else (0, 165, 255) # Hijau jika baru, Oranye jika "memori"
            # cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Terapkan overlay
            if show_overlay:
                frame = overlay_hairstyle(
                    frame, 
                    hairstyle_list[current_hairstyle_index], 
                    current_box_to_draw,
                    y_offset_factor=0.45,
                    scale_factor=1
                )
        
        # --- Langkah 3.6: Tampilkan FPS ---
        fps_frames += 1
        if time.time() - fps_start > 1.0:
            fps = fps_frames / (time.time() - fps_start)
            fps_text = f"FPS: {fps:.2f}"
            fps_frames = 0
            fps_start = time.time()
            
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Overlay: {'ON' if show_overlay else 'OFF'} (Tekan 'h')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        style_text = f"Style: {current_hairstyle_index + 1}/{len(hairstyle_list)} (Tekan 'n'/'p')"
        cv2.putText(frame, style_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Tampilkan hasil
        cv2.imshow("SVM+ORB Hairstyle Overlay", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'): # Tombol 'h' untuk toggle (hide/show)
            show_overlay = not show_overlay
        elif key == ord('n'): # 'n' untuk Next (Selanjutnya)
            current_hairstyle_index = (current_hairstyle_index + 1) % len(hairstyle_list)
        elif key == ord('p'): # 'p' untuk Previous (Sebelumnya)
            current_hairstyle_index = (current_hairstyle_index - 1) % len(hairstyle_list)
            
    # --- Selesai ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_webcam()
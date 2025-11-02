import cv2
import numpy as np
import joblib
import logging
from pathlib import Path
from tqdm import tqdm  # Untuk progress bar
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Konfigurasi Awal ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

POS_DIR = Path("data/faces")
NEG_DIR = Path("data/non_faces")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)  # Buat folder /models jika belum ada

# Konfigurasi Model
K_CLUSTERS = 512      # Jumlah "kata" dalam kamus BoVW
IMG_SIZE = (128, 128) # Ukuran standar untuk setiap crop
ORB_FEATURES = 500    # Maksimum "sidik jari" per gambar

# --- 1. Fungsi Helper untuk Fitur (Mata) ---

def create_orb_detector(n_features=ORB_FEATURES):
    """Membuat detektor ORB."""
    return cv2.ORB_create(nfeatures=n_features)

def extract_descriptors(img_path, orb, size=IMG_SIZE):
    """Mengekstrak "sidik jari" (descriptors) dari satu gambar."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Standarisasi ukuran gambar
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return descriptors
    except Exception as e:
        logging.warning(f"Gagal memproses {img_path}: {e}")
        return None

# --- 2. Fungsi Helper untuk BoVW (Kamus & Histogram) ---

def build_codebook(all_descriptors, k=K_CLUSTERS):
    """Membuat 'kamus' (codebook) dari semua sidik jari."""
    logging.info(f"Mengelompokkan {len(all_descriptors)} sidik jari menjadi {k} kamus...")
    
    # Gabungkan semua descriptor jadi satu array besar
    descriptor_stack = np.vstack(all_descriptors)
    
    # Gunakan MiniBatchKMeans untuk kecepatan
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(descriptor_stack)
    
    logging.info("Pembuatan kamus (codebook) selesai.")
    return kmeans

def compute_bovw_histogram(descriptors, codebook):
    """Menghitung 1 histogram (vektor) untuk 1 gambar."""
    k = codebook.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k, dtype=np.float32) # Vektor kosong jika tak ada fitur

    # Prediksi kata terdekat untuk setiap descriptor
    indices = codebook.predict(descriptors)
    
    # Buat histogram
    hist, _ = np.histogram(indices, bins=np.arange(k + 1))
    
    # Normalisasi (L1 norm)
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist, ord=1)
    if norm > 0:
        hist /= norm
        
    return hist

# --- 3. Fungsi Utama Training ---

def main_train():
    logging.info("--- Memulai Proses Training ---")
    orb = create_orb_detector()

    # --- Langkah 3.1: Load Path Gambar & Label ---
    logging.info("Membaca path dataset...")
    extensions = ("*.jpg", "*.jpeg", "*.png")
    pos_paths = [p for ext in extensions for p in POS_DIR.glob(ext)]
    neg_paths = [p for ext in extensions for p in NEG_DIR.glob(ext)]
    
    if not pos_paths or not neg_paths:
        logging.error("Dataset tidak ditemukan! Pastikan folder data/faces dan data/non_faces terisi.")
        return

    all_paths = pos_paths + neg_paths
    # Label: 1 untuk 'face', 0 untuk 'non_face'
    labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))

    # Bagi data untuk training & testing (80% train, 20% test)
    paths_train, paths_test, y_train, y_test = train_test_split(
        all_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logging.info(f"Total: {len(all_paths)} gambar. Train: {len(paths_train)}, Test: {len(paths_test)}")

    # --- Langkah 3.2: Ekstraksi Descriptors (HANYA dari data train) ---
    logging.info("Mengekstrak 'sidik jari' (descriptors) dari data training...")
    train_descriptors_list = []
    for path in tqdm(paths_train, desc="Ekstraksi Fitur"):
        des = extract_descriptors(path, orb)
        if des is not None:
            train_descriptors_list.append(des)

    # --- Langkah 3.3: Bangun Kamus (Codebook) ---
    codebook = build_codebook(train_descriptors_list)
    joblib.dump(codebook, MODEL_DIR / "codebook.pkl")
    logging.info(f"Kamus (codebook) disimpan di {MODEL_DIR / 'codebook.pkl'}")
    
    # --- Langkah 3.4: Buat Histogram (Vektor Fitur) ---
    logging.info("Mengubah semua gambar menjadi Vektor Histogram (BoVW)...")
    
    X_train = []
    for path in tqdm(paths_train, desc="Membuat Vektor Train"):
        des = extract_descriptors(path, orb) # Ekstrak lagi (atau simpan dari atas)
        hist = compute_bovw_histogram(des, codebook)
        X_train.append(hist)
    
    X_test = []
    for path in tqdm(paths_test, desc="Membuat Vektor Test"):
        des = extract_descriptors(path, orb)
        hist = compute_bovw_histogram(des, codebook)
        X_test.append(hist)
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # --- Langkah 3.5: Scaling Fitur ---
    logging.info("Melakukan scaling fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    logging.info(f"Scaler disimpan di {MODEL_DIR / 'scaler.pkl'}")

    # --- Langkah 3.6: Latih SVM (Hakim) ---
    logging.info("Melatih Hakim (LinearSVC)...")
    svm = LinearSVC(dual='auto', C=1.0, random_state=42, max_iter=5000)
    svm.fit(X_train_scaled, y_train)
    
    joblib.dump(svm, MODEL_DIR / "svm.pkl")
    logging.info(f"Model SVM (Hakim) disimpan di {MODEL_DIR / 'svm.pkl'}")
    
    # --- Langkah 3.7: Evaluasi Model ---
    logging.info("--- Hasil Evaluasi di Test Set ---")
    y_pred = svm.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=["non_face (0)", "face (1)"]))
    logging.info("--- Training Selesai ---")

if __name__ == "__main__":
    main_train()
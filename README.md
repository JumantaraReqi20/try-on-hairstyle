# Virtual Hairstyle Try-On untuk Barbershop

Sebuah solusi *virtual mirror* (cermin virtual) *real-time* yang memungkinkan pelanggan mencoba berbagai model rambut secara instan sebelum potong rambut.

Dikembangkan oleh: **Tim Trifur - Politeknik Negeri Bandung**


## 💈 Tentang Proyek

Proyek ini adalah sistem *virtual try-on* yang dirancang khusus untuk merevolusi pengalaman pelanggan di barbershop. Kami memahami bahwa "salah potong" adalah kekhawatiran terbesar pelanggan. Solusi kami mengatasinya dengan menyediakan "cermin ajaib" yang memungkinkan pelanggan melihat pratinjau berbagai model rambut di wajah mereka secara *real-time* menggunakan webcam.

Tujuan kami adalah untuk dikomersialisasikan kepada para pengelola barbershop sebagai alat konsultasi yang inovatif untuk meningkatkan kepuasan pelanggan, mengurangi miskomunikasi, dan menaikkan *up-selling* model rambut premium.

## ✨ Fitur Utama

* **Deteksi Wajah Real-Time:** Menggunakan pipeline *Computer Vision* klasik (Haar Cascade, ORB, dan SVM) untuk deteksi wajah yang cepat dan efisien.
* **Overlay Model Rambut:** Menempatkan aset model rambut (PNG transparan) secara stabil di atas wajah yang terdeteksi, dengan *smoothing* untuk mengurangi "getaran" (jitter).
* **Interaktivitas Penuh:** Pelanggan dapat mengganti model rambut secara instan menggunakan tombol di layar ("Next Style", "Prev Style").
* **Arsitektur Client-Server:**
    * **Backend (Python):** Melakukan semua pemrosesan *Computer Vision* yang berat (deteksi, validasi SVM, overlay).
    * **Frontend (Godot):** Menampilkan antarmuka (UI) yang ringan, interaktif, dan ramah pengguna di *booth* barbershop.
* **Streaming Laten-Rendah:** Menggunakan protokol **UDP** untuk mengirimkan *frame* video yang sudah di-overlay dari server Python ke klien Godot, memastikan pengalaman yang mulus dan nyaris tanpa jeda.

## 💻 Arsitektur & Tumpukan Teknologi

Sistem ini berjalan sebagai dua aplikasi terpisah yang berkomunikasi melalui jaringan lokal (UDP).

* **Backend (Server CV):**
    * **Python 3.10+**
    * **OpenCV:** Untuk pemrosesan gambar, deteksi Haar, dan fitur ORB.
    * **Scikit-learn:** Untuk model klasifikasi SVM yang memvalidasi deteksi wajah.
    * **Numpy & Joblib:** Untuk operasi data dan manajemen model.

* **Frontend (Klien UI):**
    * **Godot Engine 4.x**
    * **GDScript:** Untuk mengelola UI, menerima *stream* UDP, dan mengirim perintah (ganti model) kembali ke server.

## 🚀 Instalasi & Setup

Repositori ini **tidak** menyertakan dataset training atau model `.pkl` yang sudah dilatih karena ukurannya yang sangat besar. Anda harus mengunduhnya secara manual.

### 1. Prasyarat

* Python 3.10 atau lebih baru
* Godot Engine 4.3
* Kamera (Webcam)

### 2. Setup Backend (Server Python)

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/](https://github.com/)JumantaraReqi20/try-on-hairstyle.git
    cd try-on-hair-style
    ```

2.  **Buat Virtual Environment (Sangat Direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # atau venv\Scripts\activate di Windows
    ```

3.  **Instal dependensi Python:**
    Buat file `requirements.txt` yang berisi:
    ```
    opencv-python
    scikit-learn
    numpy
    joblib
    tqdm
    ```
    Lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Aset Kritis (Dataset & Model):**
    Karena file-file ini terlalu besar untuk Git, silakan unduh dari link eksternal di bawah ini:

    * **Unduh Dataset:** `---`
    * **Unduh Model:** `---`

5.  **Tempatkan Aset:**
    * Ekstrak file dataset dan pastikan strukturnya seperti ini:
        ```
        data/
        ├── faces/
        │   ├── face_001.jpg
        │   └── ...
        └── non_faces/
            ├── non_face_001.jpg
            └── ...
        ```
    * Tempatkan file model (`codebook.pkl`, `scaler.pkl`, `svm.pkl`) di dalam folder `models/`.
    * Tempatkan file aset rambut (`hairstyle_1.png`, `hairstyle_2.png`, dst.) di dalam folder `assets/`.

6.  **(Opsional) Latih Model Anda Sendiri:**
    Jika Anda ingin melatih ulang model dengan data Anda sendiri, jalankan:
    ```bash
    python train.py
    ```

### 3. Setup Frontend (Klien Godot)

1.  Buka aplikasi Godot Engine.
2.  Pilih "Impor" (Import) dan arahkan ke file `project.godot` di dalam folder proyek ini.
3.  Buka proyek tersebut. Adegan utama (`webcam_client_udp.tscn`) sudah siap digunakan.

## ▶️ Cara Menjalankan

1.  **Jalankan Server Python:**
    Pastikan webcam Anda terhubung dan tidak digunakan oleh aplikasi lain. Buka terminal di folder proyek dan jalankan:
    ```bash
    python server_svm_udp.py
    ```
    Anda akan melihat log: `🚀 Server SVM-UDP dimulai di 127.0.0.1:8888`

2.  **Jalankan Klien Godot:**
    Buka proyek di Godot Engine dan tekan **F5** (atau tombol Play) untuk menjalankan adegan utama.

3.  **Tes Aplikasi:**
    * Klik tombol **"Connect to Server"** di aplikasi Godot.
    * Video dari server Python (lengkap dengan overlay rambut) akan muncul di layar Godot.
    * Gunakan tombol **"Next Style"** dan **"Prev Style"** untuk mengganti model rambut.

## 👥 Tim Kami (Trifur - Politeknik Negeri Bandung)

Proyek ini dikembangkan dan dikelola oleh:

| Nama | NIM | Peran |
| :--- | :--- | :--- |
| **Farrel Zandra** | `231524007` | Frontend Developer - Technical Writer |
| **Reqi Jumantara Hapid** | `231524023` | Backend Developer - Project Manager |
| **Umar Faruq Robbany** | `231524028` | UI/UX Designer - Data Engineer |

## 💼 Komersialisasi

Kami saat ini sedang **aktif mengembangkan proyek ini untuk tujuan komersialisasi**.  
Untuk pertanyaan bisnis, kemitraan, atau permintaan demo, silakan hubungi kami melalui email berikut:

📧 **reqi.jumantara.tif423@polban.ac.id**.

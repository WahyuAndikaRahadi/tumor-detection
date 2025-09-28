# 🧠 Aplikasi Web Deteksi Tumor Otak

Aplikasi web untuk deteksi tumor otak menggunakan deep learning dan MRI scan dengan akurasi 99%. Dibuat oleh **Wahyu Andika Rahadi** sebagai Fullstack Developer.

## 📊 Dataset Information

Dataset yang digunakan dalam aplikasi ini:
- **Total Gambar**: 253 gambar MRI otak
- **Gambar Tumor**: 155 gambar (61.3%)
- **Gambar Normal**: 98 gambar (38.7%)
- **Format**: RGB, diresize ke 128x128 pixels

## 🌐 Fitur Multi-Bahasa

Aplikasi mendukung dua bahasa:
- 🇮🇩 **Bahasa Indonesia** (default)
- 🇺🇸 **English**

## 📋 Prerequisites

- Python 3.7 atau lebih baru
- pip (Python package manager)
- Model file `model_tumor.h5` (hasil training Anda)

## 🚀 Instalasi dan Penggunaan

### 1. Persiapan File
Pastikan Anda memiliki file-file berikut:
```
project_folder/
├── app.py                 # Aplikasi Streamlit utama
├── model_tumor.h5         # Model yang sudah di-train
├── requirements.txt       # Dependencies
└── README.md             # Dokumentasi
```

### 2. Install Dependencies
```bash
# Install semua package yang diperlukan
pip install -r requirements.txt

# Atau install satu per satu jika ada error
pip install streamlit==1.28.1
pip install tensorflow==2.13.0
pip install opencv-python==4.8.1.78
```

### 3. Jalankan Aplikasi
```bash
# Jalankan aplikasi Streamlit
streamlit run app.py

# Atau dengan port khusus
streamlit run app.py --server.port 8502
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 🎯 Fitur Aplikasi

### 🏠 Beranda & Upload
- **Upload gambar MRI**: Drag & drop atau pilih file
- **Prediksi real-time**: Menggunakan model AI dengan akurasi 99%
- **Visualisasi confidence**: Gauge chart interaktif
- **Rekomendasi medis**: Berdasarkan hasil prediksi
- **Detail gambar**: Informasi file yang diupload

### 📊 Performa Model  
- **Metrics lengkap**: Training, validation, dan test accuracy
- **Grafik training history**: Visualisasi performa training
- **Dataset statistics**: Informasi distribusi data aktual
- **Confusion matrix**: Analisis hasil prediksi

### ℹ️ Tentang Dataset
- **Informasi dataset**: 253 gambar MRI (155 tumor, 98 normal)
- **Konteks medis**: Informasi tentang tumor otak
- **Statistik detail**: Distribusi dan karakteristik data

### 🔬 Detail Teknis
- **Arsitektur CNN**: Detail layer dan parameter model
- **Konfigurasi training**: Optimizer, loss function, dll
- **System requirements**: Hardware dan software
- **Deployment guide**: Panduan instalasi dan deployment

## 📊 Spesifikasi Model

- **Arsitektur**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128x3 (gambar RGB)  
- **Total Parameter**: 3,304,769 (12.61 MB)
- **Training Accuracy**: 99.0%
- **Test Accuracy**: 90.2%
- **Waktu Prediksi**: <1 detik

## 🏥 Medical Disclaimer

⚠️ **PENTING**: Aplikasi ini hanya untuk tujuan edukasi dan penelitian. Selalu konsultasikan dengan dokter spesialis untuk diagnosis medis yang akurat. Hasil AI tidak menggantikan judgement medis profesional.

## 🔧 Troubleshooting

### Model File Not Found
```
❌ File model 'model_tumor.h5' tidak ditemukan
```
**Solusi**: Pastikan file `model_tumor.h5` ada di folder yang sama dengan `app.py`

### TensorFlow Version Issues
```
❌ Error loading model: incompatible model version
```
**Solusi**: 
```bash
pip install tensorflow==2.13.0
# Atau coba versi lain yang kompatibel
pip install tensorflow==2.12.0
```

### Memory Issues
```
❌ OOM (Out of Memory) error
```
**Solusi**: 
- Restart aplikasi jika memory usage tinggi
- Gunakan gambar dengan ukuran file lebih kecil
- Close tab browser lain yang tidak diperlukan

### Dependencies Error
```bash
# Update semua packages
pip install --upgrade -r requirements.txt

# Atau install ulang dari scratch
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### OpenCV Error (Linux)
```bash
# Install sistem dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

## 📱 Tips Penggunaan

### Untuk Upload Gambar:
- **Format yang didukung**: PNG, JPG, JPEG
- **Ukuran**: Tidak ada batasan (otomatis diresize ke 128x128)
- **Kualitas**: Gunakan MRI scan dengan kualitas tinggi
- **Orientasi**: Pastikan gambar dalam orientasi yang benar

### Untuk Interpretasi Hasil:
- **Confidence >90%**: Hasil sangat dapat dipercaya
- **Confidence 70-90%**: Hasil cukup dapat dipercaya  
- **Confidence <70%**: Perlu verifikasi tambahan dengan dokter

## 🚀 Deployment Options

### 1. Local Development
```bash
streamlit run app.py --server.port 8501
```

### 2. Streamlit Cloud
1. Upload code ke GitHub repository
2. Connect ke [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy dengan satu klik

### 3. Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless", "true", "--server.enableCORS", "false"]
```

```bash
# Build dan run Docker container
docker build -t brain-tumor-app .
docker run -p 8501:8501 brain-tumor-app
```

### 4. Heroku Deployment
```bash
# Install Heroku CLI dan login
heroku create your-app-name
git add .
git commit -m "Deploy brain tumor detection app"
git push heroku main
```

Tambahkan file `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.headless true
```

## 📞 Support & Contact

Jika mengalami masalah atau butuh bantuan:

1. **Check dokumentasi** ini terlebih dahulu
2. **Verify dependencies** dan Python version
3. **Pastikan model file** sudah ada dan tidak corrupt
4. **Check system resources** (RAM, storage)

**Developer Contact:**
- **Nama**: Wahyu Andika Rahadi
- **Role**: Fullstack Developer  
- **Specialization**: AI/ML Applications, Web Development

## 📄 License & Credits

Aplikasi ini dibuat untuk tujuan edukasi dan penelitian. Silakan gunakan dan modifikasi sesuai kebutuhan.

**Technologies Used:**
- **Frontend**: Streamlit, HTML/CSS, JavaScript
- **Backend**: Python, TensorFlow  
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Image Processing**: OpenCV, PIL

---

**💻 Created with ❤️ by Wahyu Andika Rahadi as Fullstack Developer**

*Aplikasi Deteksi Tumor Otak - Membantu tenaga medis dengan teknologi AI*
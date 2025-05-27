# Dashboard Analisis Risiko & Segmentasi Diabetes

Dashboard interaktif untuk analisis risiko diabetes dan segmentasi pasien menggunakan model machine learning.

## Fitur

- Prediksi risiko diabetes menggunakan model Regresi Logistik
- Segmentasi pasien menggunakan K-Means Clustering
- Visualisasi data dan hasil analisis
- Antarmuka pengguna yang interaktif

## Teknologi yang Digunakan

- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Cara Menjalankan Dashboard

1. Clone repository ini:
```bash
git clone [URL_REPOSITORY_ANDA]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan dashboard:
```bash
streamlit run app.py
```

## Struktur File

- `app.py`: File utama dashboard Streamlit
- `requirements.txt`: Daftar dependencies
- `logistic_model.joblib`: Model Regresi Logistik yang sudah dilatih
- `scaler.joblib`: Scaler untuk normalisasi data

## Deployment

Dashboard ini dapat diakses melalui Streamlit Cloud setelah di-deploy dari repository GitHub.

## Kontributor

Kelompok 2 SI4706 - Tugas Besar Penambangan Data 
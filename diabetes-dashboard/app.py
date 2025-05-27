import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             roc_auc_score, silhouette_score)
from joblib import load
# KMeans dari sklearn.cluster mungkin tidak perlu diimpor jika hanya load, tapi tidak apa-apa jika ada
from sklearn.cluster import KMeans

# Initialize session state for active tab (opsional, bisa dihapus jika tidak digunakan aktif)
# if 'active_tab' not in st.session_state:
#     st.session_state.active_tab = "Gambaran Umum"

# Set page config
st.set_page_config(
    page_title="Dashboard Prediksi dan Segmentasi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

# --- DATA LOADING, PREPARATION, AND MODEL LOADING ---
@st.cache_resource # Cache untuk resource yang berat seperti model
def load_all_assets():
    log_reg = load('logistic_model.joblib')
    scaler = load('scaler.joblib')
    kmeans_model = load('kmeans_model.joblib') # MUAT MODEL K-MEANS DARI FILE

    # Muat dan proses data HANYA SEKALI
    # Ganti dengan path lokal jika file CSV disertakan dalam deployment untuk kecepatan lebih
    # Misalnya: df_initial = pd.read_csv('diabetes.csv')
    df_initial = pd.read_csv('https://raw.githubusercontent.com/ulhaqdhifulloh/TUBES-DATMIN-TEAM2-SI4706/main/diabetes.csv')
    
    df_processed = df_initial.copy() # Kita akan memproses dataframe ini
    
    # Handle missing values (nilai nol yang tidak valid)
    cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed[cols_with_zero_invalid] = df_processed[cols_with_zero_invalid].replace(0, np.nan)
    
    # Isi NaN (sesuai strategi di notebook: mean untuk Glucose, BloodPressure; median untuk lainnya)
    df_processed['Glucose'].fillna(df_processed['Glucose'].mean(), inplace=True)
    df_processed['BloodPressure'].fillna(df_processed['BloodPressure'].mean(), inplace=True)
    df_processed['SkinThickness'].fillna(df_processed['SkinThickness'].median(), inplace=True)
    df_processed['Insulin'].fillna(df_processed['Insulin'].median(), inplace=True)
    df_processed['BMI'].fillna(df_processed['BMI'].median(), inplace=True)
    
    # Definisikan kolom fitur (HARUS 8 FITUR, SESUAI SAAT TRAINING SCALER DAN SEMUA MODEL)
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Pisahkan fitur dan target sebelum scaling
    X_for_scaling = df_processed[feature_cols]
    y_outcome = df_processed['Outcome'].copy() # Salin untuk menghindari SettingWithCopyWarning

    # Scaling fitur menggunakan scaler yang sudah di-load
    # scaler.transform() mengembalikan NumPy array
    X_scaled_np = scaler.transform(X_for_scaling)
    
    # Simpan X_scaled_np sebagai NumPy array untuk konsistensi input ke model
    # Jika model dilatih dengan nama fitur, pastikan konsisten atau latih tanpa nama fitur
    # Untuk K-Means dan Logistic Regression dari scikit-learn, NumPy array umumnya aman
    
    # df_processed akan digunakan untuk tampilan data mentah/deskriptif
    # y_outcome untuk evaluasi
    # X_scaled_np untuk prediksi dan evaluasi performa model pada keseluruhan data
    return log_reg, scaler, kmeans_model, X_scaled_np, y_outcome, df_processed, feature_cols

# Load semua aset (model, scaler, data yang diproses dan di-scale)
log_reg, scaler, kmeans_model, X_full_scaled, y_full, df_display, feature_columns = load_all_assets()
# X_full_scaled adalah NumPy array dengan 8 fitur yang sudah di-scaling
# y_full adalah Series Pandas untuk Outcome
# df_display adalah DataFrame Pandas yang sudah diproses (NaN handling) tapi belum di-scale, bisa untuk tampilan umum

# --- UI LAYOUT ---
st.title("Dashboard Interaktif: Analisis Risiko & Segmentasi Diabetes Pasien Pima Indian")
st.markdown("""
Selamat datang di dashboard analisis risiko diabetes. Dashboard ini bertujuan untuk:
- Membantu identifikasi dini pasien yang berisiko tinggi mengidap diabetes.
- Melakukan segmentasi pasien berdasarkan karakteristik medis untuk pemahaman yang lebih baik.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🏠 Gambaran Umum & Info Dasar", 
    "🩸 Prediksi Risiko Diabetes", 
    "⚕️ Segmentasi Pasien & Karakteristik"
])

# --- TAB 1: Gambaran Umum & Info Dasar ---
with tab1:
    st.header("Gambaran Umum Proyek")
    
    st.subheader("A. Pemahaman Bisnis (Business Understanding)")
    st.markdown("""
    Tujuan utama dari proyek ini adalah untuk membantu tenaga medis atau institusi kesehatan dalam 
    mengidentifikasi pasien yang berisiko tinggi mengidap diabetes. Dengan prediksi dini berdasarkan 
    data medis, tindakan pencegahan atau penanganan lebih lanjut dapat dilakukan secara proaktif, 
    sehingga dapat menurunkan risiko komplikasi kesehatan yang lebih serius di kemudian hari.
    """)
    
    st.subheader("B. Pemahaman Data (Data Understanding) & Analisis Data Awal")
    st.markdown("""
    Sumber Data: Dataset berasal dari *National Institute of Diabetes and Digestive and Kidney Diseases* dan tersedia secara publik melalui platform Kaggle. Data ini hanya mencakup perempuan berusia ≥21 tahun 
    dari suku Pima Indian.
    """)
    
    st.markdown("**Distribusi Pasien Diabetes dalam Dataset:**")
    outcome_counts = df_display['Outcome'].value_counts() # Gunakan df_display
    fig_outcome, ax_outcome = plt.subplots(figsize=(6,4))
    outcome_counts.plot(kind='pie', ax=ax_outcome, autopct='%1.1f%%', startangle=90, 
                        labels=['Tidak Diabetes (0)', 'Diabetes (1)'], colors=['skyblue', 'salmon'])
    ax_outcome.set_ylabel('')
    st.pyplot(fig_outcome)
    
    with st.expander("Lihat Statistik Deskriptif & Contoh Data"):
        st.markdown("**Statistik Deskriptif Fitur (Sebelum Scaling):**")
        st.dataframe(df_display[feature_columns].describe()) # Gunakan df_display
        st.markdown("**Contoh Data (5 Baris Pertama, Sebelum Scaling):**")
        st.dataframe(df_display.head()) # Gunakan df_display
    
    st.subheader("C. Sekilas Info Model (Model Info - Gambaran Umum)")
    # Ambil n_clusters dari model K-Means yang di-load jika tersedia
    n_clusters_kmeans = kmeans_model.n_clusters if hasattr(kmeans_model, 'n_clusters') else "N/A"
    st.markdown(f"""
    - **Prediksi Risiko:** Menggunakan model **Regresi Logistik**.
    - **Segmentasi Pasien:** Menggunakan algoritma **K-Means Clustering** (dengan {n_clusters_kmeans} segmen).
    """)


# --- TAB 2: Prediksi Risiko Diabetes ---
with tab2:
    st.header("Prediksi Risiko Diabetes (Regresi Logistik)")
    
    st.subheader("A. Lakukan Prediksi Interaktif")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        # ... (Input fields tetap sama) ...
        with col1:
            pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1, help="Jumlah berapa kali hamil.")
            glucose = st.number_input("Glukosa Plasma (mg/dL)", min_value=0, max_value=300, value=120, help="Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral.")
            blood_pressure = st.number_input("Tekanan Darah Diastolik (mm Hg)", min_value=0, max_value=150, value=70, help="Tekanan darah diastolik.")
            skin_thickness = st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", min_value=0, max_value=100, value=20, help="Ketebalan lipatan kulit trisep.")
        with col2:
            insulin = st.number_input("Insulin Serum 2 Jam (mu U/ml)", min_value=0, max_value=1000, value=80, help="Insulin serum 2 jam.")
            bmi = st.number_input("Indeks Massa Tubuh (BMI) (kg/m²)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f", help="Indeks massa tubuh.")
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.470, format="%.3f", help="Fungsi silsilah diabetes, skor yang mewakili riwayat keluarga.")
            age = st.number_input("Usia (tahun)", min_value=21, max_value=100, value=33, help="Usia pasien.")
        submit_button = st.form_submit_button(label="Prediksi Risiko Diabetes")

    if submit_button:
        # Buat DataFrame dari input pengguna
        input_data_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                       insulin, bmi, diabetes_pedigree, age]], 
                                     columns=feature_columns)
        # Scaling input pengguna menggunakan scaler yang sudah di-load
        input_scaled_np = scaler.transform(input_data_df) # Hasilnya NumPy array
        
        # Prediksi menggunakan model Regresi Logistik
        # Model scikit-learn bisa menerima NumPy array jika dilatih dengan cara yang sama
        prediction_proba = log_reg.predict_proba(input_scaled_np)[0][1]
        prediction = log_reg.predict(input_scaled_np)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi untuk Pasien:")
        if prediction == 1:
            st.error(f"Pasien diprediksi: **BERISIKO DIABETES** 😢 (Probabilitas: {prediction_proba:.2f})")
        else:
            st.success(f"Pasien diprediksi: **TIDAK BERISIKO DIABETES** 😊 (Probabilitas Risiko: {prediction_proba:.2f})")
        st.markdown("---")

    st.subheader("B. Analisis Model Prediksi")
    st.markdown("**Faktor Risiko Utama (Feature Importance):**")
    if hasattr(log_reg, 'coef_'):
        feature_importance_df = pd.DataFrame({
            'Faktor': feature_columns,
            'Pengaruh (Koefisien Regresi Logistik)': log_reg.coef_[0]
        }).sort_values(by='Pengaruh (Koefisien Regresi Logistik)', ascending=False)
        
        fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
        # Hapus hue='Faktor' jika menyebabkan error atau tidak diinginkan, legend=False sudah ada
        sns.barplot(data=feature_importance_df, x='Pengaruh (Koefisien Regresi Logistik)', y='Faktor', 
                    ax=ax_importance, palette="viridis") # legend=False dihapus karena tidak ada hue
        ax_importance.set_title('Pengaruh Faktor Risiko terhadap Prediksi Diabetes')
        st.pyplot(fig_importance)
    else:
        st.warning("Informasi feature importance (koefisien) tidak tersedia untuk model ini.")


    with st.expander("Info & Performa Model Regresi Logistik (Model Info)"):
        st.markdown("**Model:** Regresi Logistik")
        
        # Evaluasi pada keseluruhan data yang sudah di-load dan di-scale
        y_pred_log_full = log_reg.predict(X_full_scaled) # X_full_scaled adalah NumPy array
        
        accuracy = accuracy_score(y_full, y_pred_log_full)
        precision = precision_score(y_full, y_pred_log_full, zero_division=0)
        recall = recall_score(y_full, y_pred_log_full, zero_division=0)
        f1 = f1_score(y_full, y_pred_log_full, zero_division=0)
        
        st.markdown(f"- **Akurasi:** {accuracy:.3f}")
        st.markdown(f"- **Presisi:** {precision:.3f}")
        st.markdown(f"- **Recall:** {recall:.3f}")
        st.markdown(f"- **F1-score:** {f1:.3f}")

        st.markdown("**Confusion Matrix (pada keseluruhan data):**")
        cm = confusion_matrix(y_full, y_pred_log_full)
        fig_cm, ax_cm = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['Tidak Diabetes', 'Diabetes'], 
                    yticklabels=['Tidak Diabetes', 'Diabetes'])
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

# --- TAB 3: Segmentasi Pasien & Karakteristik ---
with tab3:
    st.header("Segmentasi Pasien & Karakteristik (K-Means Clustering)")
    
    # Dapatkan label cluster dari model K-Means yang sudah di-load
    # kmeans_model harusnya bisa predict dari X_full_scaled (NumPy array 8 fitur scaled)
    cluster_labels_full = kmeans_model.predict(X_full_scaled) 
    
    # Buat DataFrame baru untuk tampilan di tab ini, gabungkan fitur asli dengan cluster
    # Ambil fitur asli dari df_display (sebelum scaling) agar mudah diinterpretasi
    df_tab3_analysis = df_display.copy() # Mulai dari data yang sudah bersih (NaN handling)
    df_tab3_analysis['Cluster'] = cluster_labels_full

    st.subheader("A. Visualisasi Segmen Pasien (PCA)")
    # PCA di-fit pada data yang sudah di-scaling (X_full_scaled)
    pca = PCA(n_components=2)
    X_pca_components = pca.fit_transform(X_full_scaled) # X_full_scaled adalah NumPy array
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    scatter = ax_pca.scatter(X_pca_components[:, 0], X_pca_components[:, 1], c=cluster_labels_full, cmap='viridis', alpha=0.7)
    ax_pca.set_xlabel('Principal Component 1')
    ax_pca.set_ylabel('Principal Component 2')
    ax_pca.set_title('Segmentasi Pasien Pima Indian (Visualisasi PCA)')
    try:
        legend1 = ax_pca.legend(*scatter.legend_elements(), title="Clusters")
        ax_pca.add_artist(legend1)
    except (AttributeError, TypeError): # Menangkap TypeError juga jika legend_elements() mengembalikan format tak terduga
        plt.colorbar(scatter, ax=ax_pca, label='Cluster')
    st.pyplot(fig_pca)

    st.subheader("B. Analisis Data & Fitur per Segmen")
    st.markdown("**Karakteristik Rata-Rata per Segmen (Fitur Asli Sebelum Scaling):**")
    # Group by pada df_tab3_analysis yang berisi fitur asli dan label cluster
    cluster_stats_original_features = df_tab3_analysis.groupby('Cluster')[feature_columns].mean()
    st.dataframe(cluster_stats_original_features)

    st.markdown("**Visualisasi Karakteristik Rata-Rata per Segmen (Fitur Asli):**")
    fig_bar_stats, ax_bar_stats = plt.subplots(figsize=(12, 7))
    cluster_stats_original_features.T.plot(kind='bar', ax=ax_bar_stats, colormap="viridis")
    ax_bar_stats.set_title('Perbandingan Karakteristik Rata-rata antar Segmen (Fitur Asli)')
    ax_bar_stats.set_ylabel('Nilai Rata-rata Fitur')
    ax_bar_stats.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar_stats)

    with st.expander("Interpretasi Segmen & Info Model K-Means (Model Info)"):
        n_clusters_kmeans_tab3 = kmeans_model.n_clusters if hasattr(kmeans_model, 'n_clusters') else "N/A"
        st.markdown(f"**Model:** K-Means Clustering (dengan {n_clusters_kmeans_tab3} segmen)")
        
        # Silhouette score dihitung pada data yang di-scaling
        sil_score = silhouette_score(X_full_scaled, cluster_labels_full)
        st.markdown(f"- **Silhouette Score (pada data scaled):** {sil_score:.3f} (Semakin mendekati 1, semakin baik pemisahannya)")
        
        st.markdown("**Interpretasi Segmen (Contoh - Sesuaikan dengan Analisis Anda):**")
        # Contoh interpretasi otomatis berdasarkan jumlah cluster
        if isinstance(n_clusters_kmeans_tab3, int):
            for i in range(n_clusters_kmeans_tab3):
                st.markdown(f"- **Segmen {i}:** Cenderung memiliki [Deskripsikan karakteristik segmen {i} di sini berdasarkan `cluster_stats_original_features`].")
        else:
            st.markdown("- **Segmen 0:** Cenderung memiliki [karakteristik dominan ...].")
            st.markdown("- **Segmen 1:** Cenderung memiliki [karakteristik dominan ...].")
        st.markdown("*Silakan sesuaikan deskripsi ini berdasarkan hasil analisis mendalam dari `cluster_stats_original_features`.*")


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
    Dashboard ini dibuat untuk tujuan demonstrasi analisis data dan tidak menggantikan diagnosis medis profesional.<br>
    Pengembangan oleh: Kelompok 2 SI4706 - Tugas Besar Penambangan Data
    </p>
    """, unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, silhouette_score)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import load

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi dan Segmentasi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Kolom yang nilai nol dianggap invalid
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Fungsi untuk memuat data dan model
def load_all_assets():
    # Muat model dan scaler
    log_reg = load('logistic_model.joblib')
    scaler = load('scaler.joblib')

    # Muat data
    df = pd.read_csv(
        'https://raw.githubusercontent.com/ulhaqdhifulloh/TUBES-DATMIN-TEAM2-SI4706/main/diabetes.csv'
    )
    # Simpan salinan sebelum preprocessing
    df_original = df.copy()

    # Ganti nilai 0 dengan NaN pada kolom tertentu
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # Tangani missing values
    df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
    df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
    df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
    df['BMI'].fillna(df['BMI'].median(), inplace=True)

    # Tangani duplikasi
    df.drop_duplicates(inplace=True)

    # Tangani outlier menggunakan IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Definisikan fitur dan target
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[feature_cols]
    y = df['Outcome']

    # Standarisasi fitur
    X_scaled = scaler.transform(X)

    # Model KMeans
    kmeans_model = KMeans(n_clusters=2, init='k-means++', n_init=10,
                        max_iter=300, random_state=42)
    kmeans_model.fit(X_scaled)

    # Kembalikan semua aset
    return log_reg, scaler, kmeans_model, X_scaled, y, df, feature_cols, df_original

# Muat semua aset
everything = load_all_assets()
log_reg, scaler, kmeans_model, X_scaled, y, df, feature_cols, df_original = everything

# Judul utama
st.title("🩺 Dashboard Interaktif: Analisis Risiko & Segmentasi Diabetes Pasien Pima Indian")
st.markdown("""
Selamat datang di dashboard analisis risiko diabetes. Dashboard ini bertujuan untuk:
- Membantu identifikasi dini pasien yang berisiko tinggi mengidap diabetes.
- Melakukan segmentasi pasien berdasarkan karakteristik medis untuk pemahaman yang lebih baik.
""")

# Definisi tab
tabs = st.tabs([
    "📊 Business Understanding",
    "📈 Data Understanding",
    "🔍 Exploratory Data Analysis",
    "🛠️ Data Preprocessing",
    "🤖 Modeling & Evaluation",
    "🧪 Prediksi Risiko Diabetes"
])

# --- Tab 1: Business Understanding ---
with tabs[0]:
    st.header("📊 Business Understanding")
    st.subheader("1. Define Business Objectives")
    st.markdown("""
    Tujuan utama dari proyek ini adalah untuk membantu tenaga medis atau institusi kesehatan dalam
    mengidentifikasi pasien yang berisiko tinggi mengidap diabetes. Dengan prediksi yang akurat,
    intervensi dini dapat dilakukan untuk mencegah komplikasi lebih lanjut.
    """)

    st.subheader("2. Assess Current Situation")
    st.markdown("""
    Saat ini, diagnosis diabetes seringkali dilakukan setelah gejala muncul, yang dapat terlambat untuk
    pencegahan. Dengan memanfaatkan data historis dan teknik pembelajaran mesin, kita dapat
    mengidentifikasi pola yang menunjukkan risiko diabetes sebelum gejala muncul.
    """)

    st.subheader("3. Formulate Data Mining Problem")
    st.markdown("""
    Merumuskan masalah sebagai klasifikasi biner: apakah seorang pasien berisiko mengidap diabetes
    (1) atau tidak (0), berdasarkan fitur medis yang tersedia.
    """)

    st.subheader("4. Determine Project Objectives")
    st.markdown("""
    - Membangun model prediksi risiko diabetes dengan akurasi tinggi.
    - Melakukan segmentasi pasien untuk memahami kelompok risiko yang berbeda.
    - Menyediakan dashboard interaktif untuk eksplorasi data dan prediksi.
    """)

    st.subheader("5. Plan Project")
    st.markdown("""
    - **Tahap 1:** Pengumpulan dan pemahaman data.
    - **Tahap 2:** Eksplorasi dan visualisasi data.
    - **Tahap 3:** Pra-pemrosesan data.
    - **Tahap 4:** Pembangunan dan evaluasi model.
    - **Tahap 5:** Implementasi dashboard interaktif.
    """)

# --- Tab 2: Data Understanding ---
with tabs[1]:
    st.header("📈 Data Understanding")
    st.markdown("Dataset yang digunakan adalah Pima Indians Diabetes Dataset yang berisi informasi medis dari 768 wanita Pima Indian berusia minimal 21 tahun.")

    st.subheader("1. Struktur Dataset")
    st.write(f"Jumlah baris dan kolom: {df.shape}")

    st.subheader("2. Informasi Dataset")
    st.dataframe(df.head())

    st.subheader("3. Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("4. Cek Missing Values")
    st.write(df.isnull().sum())

    st.subheader("5. Cek Duplikasi Data")
    st.write(f"Jumlah data duplikat: {df.duplicated().sum()}")

# --- Tab 3: Exploratory Data Analysis ---
with tabs[2]:
    st.header("🔍 Exploratory Data Analysis")
    st.subheader("1. Korelasi antar fitur")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("2. Pair Plot")
    sns.pairplot(df[feature_cols + ['Outcome']], hue='Outcome')
    st.pyplot()

    st.subheader("3. Distribusi fitur penting")
    cols_to_plot = ['Age', 'Glucose', 'BMI']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col], kde=True, ax=axs[i])
        axs[i].set_title(f'Distribusi {col}')
    st.pyplot(fig)

# --- Tab 4: Data Preprocessing ---
with tabs[3]:
    st.header("🛠️ Data Preprocessing")
    st.markdown("Berikut adalah ringkasan proses pra-pemrosesan yang dilakukan:")
    st.markdown("""
    - Mengganti nilai 0 pada kolom medis yang tidak valid menjadi NaN
    - Mengisi nilai NaN menggunakan mean atau median
    - Normalisasi fitur menggunakan `StandardScaler`
    """)

    st.subheader("1. Penanganan Missing Values")
    st.markdown("""
    Nilai 0 pada kolom tertentu dianggap sebagai missing values dan telah digantikan dengan nilai mean atau median yang sesuai.
    """)

    st.subheader("2. Penanganan Duplikasi Data")
    st.markdown("Data duplikat telah dihapus untuk memastikan integritas data.")

    st.subheader("3. Penanganan Outlier")
    st.markdown("Outlier telah diidentifikasi dan dihapus menggunakan metode IQR.")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df[feature_cols], ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    st.subheader("Sebelum dan Sesudah Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Sebelum:")
        st.dataframe(df_original[zero_cols].head())
    with col2:
        st.markdown("Sesudah:")
        st.dataframe(df[zero_cols].head())

# --- Tab 5: Modeling & Evaluation ---
with tabs[4]:
    st.header("🤖 Modeling & Evaluation")
    y_pred = log_reg.predict(X_scaled)
    st.subheader("1. Logistic Regression")
    st.markdown(f"- **Akurasi:** {accuracy_score(y, y_pred):.3f}")
    st.markdown(f"- **Presisi:** {precision_score(y, y_pred):.3f}")
    st.markdown(f"- **Recall:** {recall_score(y, y_pred):.3f}")
    st.markdown(f"- **F1-score:** {f1_score(y, y_pred):.3f}")
    cm = confusion_matrix(y, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax4)
    st.pyplot(fig4)

    st.subheader("2. K-Means Clustering")
    cluster_labels = kmeans_model.predict(X_scaled)
    df['Cluster'] = cluster_labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids = pca.transform(kmeans_model.cluster_centers_)
    fig5, ax5 = plt.subplots()
    colors = ['purple', 'green']
    for cluster in range(2):
        ax5.scatter(
            X_pca[cluster_labels == cluster, 0],
            X_pca[cluster_labels == cluster, 1],
            s=50,
            c=colors[cluster],
            label=f'Cluster {cluster}'
        )
    ax5.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroid')
    ax5.set_title('PCA Clustering with Centroids')
    ax5.legend()
    st.pyplot(fig5)

    st.subheader("Karakteristik per Cluster")
    st.dataframe(df.groupby('Cluster')[feature_cols].mean())

    fig6, ax6 = plt.subplots()
    df.groupby('Cluster')[feature_cols].mean().T.plot(kind='bar', ax=ax6)
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    df['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax7)
    ax7.set_title("Jumlah Pasien per Cluster")
    st.pyplot(fig7)

    st.subheader("Silhouette Score")
    sil = silhouette_score(X_scaled, cluster_labels)
    st.markdown(f"- **Silhouette Score:** {sil:.3f}")

# --- Tab 6: Prediksi Risiko Diabetes ---
with tabs[5]:
    st.header("🧪 Prediksi Risiko Diabetes")
    st.subheader("Masukkan Data Pasien untuk Prediksi")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, 1)
            Glucose = st.number_input("Glukosa", 0, 200, 120)
            BloodPressure = st.number_input("Tekanan Darah", 0, 140, 70)
            SkinThickness = st.number_input("Ketebalan Kulit", 0, 100, 20)
        with col2:
            Insulin = st.number_input("Insulin", 0, 600, 80)
            BMI = st.number_input("BMI", 0.0, 60.0, 25.0)
            DiabetesPedigreeFunction = st.number_input("DPF", 0.0, 3.0, 0.5)
            Age = st.number_input("Umur", 0, 120, 33)

        submitted = st.form_submit_button("🔍 Prediksi Risiko Diabetes")

    if submitted:
        # Susun data input sesuai urutan fitur
        input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                SkinThickness, Insulin, BMI,
                                DiabetesPedigreeFunction, Age]])
        # Standarisasi
        input_scaled = scaler.transform(input_data)
        # Prediksi
        prediction = log_reg.predict(input_scaled)
        prediction_proba = log_reg.predict_proba(input_scaled)[0][1]
        cluster_pred = kmeans_model.predict(input_scaled)[0]

        st.subheader("🔎 Hasil Prediksi")
        if prediction[0] == 1:
            st.error(f"⚠️ Pasien diprediksi **BERISIKO** mengidap diabetes dengan probabilitas {prediction_proba:.2%}.")
        else:
            st.success(f"✅ Pasien diprediksi **TIDAK BERISIKO** mengidap diabetes dengan probabilitas {prediction_proba:.2%}.")

        st.subheader("👥 Segmentasi Pasien")
        st.info(f"Pasien tergolong dalam **Cluster {cluster_pred}**, dengan karakteristik mirip kelompok pasien lainnya dalam cluster ini.")

        st.markdown("📌 *Catatan: Prediksi ini hanya sebagai alat bantu dan bukan pengganti diagnosis profesional.*")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>© 2025 - Kelompok 2 SI4706 - Tugas Besar Data Mining</p>",
    unsafe_allow_html=True
)

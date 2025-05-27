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
from sklearn.cluster import KMeans

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Gambaran Umum"

# Set page config
st.set_page_config(
    page_title="Dashboard Prediksi dan Segmentasi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

# --- DATA LOADING AND PREPARATION ---
@st.cache_resource
def load_models_and_scaler():
    log_reg = load('logistic_model.joblib')
    scaler = load('scaler.joblib')
    df_initial = pd.read_csv('https://raw.githubusercontent.com/ulhaqdhifulloh/TUBES-DATMIN-TEAM2-SI4706/main/diabetes.csv')
    df_initial_copy = df_initial.copy()
    cols_with_zero_invalid_initial = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_initial_copy[cols_with_zero_invalid_initial] = df_initial_copy[cols_with_zero_invalid_initial].replace(0, np.nan)
    df_initial_copy['Glucose'].fillna(df_initial_copy['Glucose'].mean(), inplace=True)
    df_initial_copy['BloodPressure'].fillna(df_initial_copy['BloodPressure'].mean(), inplace=True)
    df_initial_copy['SkinThickness'].fillna(df_initial_copy['SkinThickness'].median(), inplace=True)
    df_initial_copy['Insulin'].fillna(df_initial_copy['Insulin'].median(), inplace=True)
    df_initial_copy['BMI'].fillna(df_initial_copy['BMI'].median(), inplace=True)
    
    feature_cols_initial = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_initial_scaled = scaler.transform(df_initial_copy[feature_cols_initial])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans.fit(X_initial_scaled)
    return log_reg, scaler, kmeans, X_initial_scaled, df_initial_copy['Outcome']

@st.cache_data
def load_processed_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ulhaqdhifulloh/TUBES-DATMIN-TEAM2-SI4706/main/diabetes.csv')
    df_copy = df.copy()
    
    cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_copy[cols_with_zero_invalid] = df_copy[cols_with_zero_invalid].replace(0, np.nan)
    
    df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
    df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
    df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
    df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
    df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)
    
    return df_copy

# Load models, scaler, fitted k-means, and full scaled data
log_reg, scaler, kmeans, X_full_scaled, y_full = load_models_and_scaler()
df_processed = load_processed_data()

feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- UI LAYOUT ---
st.title("Dashboard Interaktif: Analisis Risiko & Segmentasi Diabetes Pasien Pima Indian")
st.markdown("""
Selamat datang di dashboard analisis risiko diabetes. Dashboard ini bertujuan untuk:
- Membantu identifikasi dini pasien yang berisiko tinggi mengidap diabetes.
- Melakukan segmentasi pasien berdasarkan karakteristik medis untuk pemahaman yang lebih baik.
""")

# Function to handle tab changes
def on_tab_change(tab_name):
    st.session_state.active_tab = tab_name

# Create tabs with session state
tab1, tab2, tab3 = st.tabs([
    "🏠 Gambaran Umum & Info Dasar", 
    "🩸 Prediksi Risiko Diabetes", 
    "⚕️ Segmentasi Pasien & Karakteristik"
])

# --- TAB 1: Gambaran Umum & Info Dasar ---
with tab1:
    on_tab_change("Gambaran Umum")
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
    
    # Distribusi Outcome
    st.markdown("**Distribusi Pasien Diabetes dalam Dataset:**")
    outcome_counts = df_processed['Outcome'].value_counts()
    fig_outcome, ax_outcome = plt.subplots(figsize=(6,4))
    outcome_counts.plot(kind='pie', ax=ax_outcome, autopct='%1.1f%%', startangle=90, 
                        labels=['Tidak Diabetes (0)', 'Diabetes (1)'], colors=['skyblue', 'salmon'])
    ax_outcome.set_ylabel('')
    st.pyplot(fig_outcome)
    
    with st.expander("Lihat Statistik Deskriptif & Contoh Data"):
        st.markdown("**Statistik Deskriptif Fitur:**")
        st.dataframe(df_processed[feature_columns].describe())
        st.markdown("**Contoh Data (5 Baris Pertama):**")
        st.dataframe(df_processed.head())
    
    st.subheader("C. Sekilas Info Model (Model Info - Gambaran Umum)")
    st.markdown("""
    - **Prediksi Risiko:** Menggunakan model **Regresi Logistik**.
    - **Segmentasi Pasien:** Menggunakan algoritma **K-Means Clustering**.
    """)

# --- TAB 2: Prediksi Risiko Diabetes ---
with tab2:
    on_tab_change("Prediksi")
    st.header("Prediksi Risiko Diabetes (Regresi Logistik)")
    
    st.subheader("A. Lakukan Prediksi Interaktif")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
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
        input_features = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                      insulin, bmi, diabetes_pedigree, age]], 
                                    columns=feature_columns)
        input_scaled = scaler.transform(input_features)
        # Convert back to DataFrame with feature names
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)
        
        prediction_proba = log_reg.predict_proba(input_scaled_df)[0][1]
        prediction = log_reg.predict(input_scaled_df)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi untuk Pasien:")
        if prediction == 1:
            st.error(f"Pasien diprediksi: **BERISIKO DIABETES** 😢 (Probabilitas: {prediction_proba:.2f})")
        else:
            st.success(f"Pasien diprediksi: **TIDAK BERISIKO DIABETES** 😊 (Probabilitas Risiko: {prediction_proba:.2f})")
        st.markdown("---")

    st.subheader("B. Analisis Model Prediksi")
    st.markdown("**Faktor Risiko Utama (Feature Importance):**")
    feature_importance = pd.DataFrame({
        'Faktor': feature_columns,
        'Pengaruh (Koefisien Regresi Logistik)': log_reg.coef_[0]
    }).sort_values(by='Pengaruh (Koefisien Regresi Logistik)', ascending=False)
    
    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Pengaruh (Koefisien Regresi Logistik)', y='Faktor', 
                ax=ax_importance, hue='Faktor', palette="viridis", legend=False)
    ax_importance.set_title('Pengaruh Faktor Risiko terhadap Prediksi Diabetes')
    st.pyplot(fig_importance)

    with st.expander("Info & Performa Model Regresi Logistik (Model Info)"):
        st.markdown("**Model:** Regresi Logistik")
        
        y_pred_full = log_reg.predict(X_full_scaled)
        
        accuracy = accuracy_score(y_full, y_pred_full)
        precision = precision_score(y_full, y_pred_full)
        recall = recall_score(y_full, y_pred_full)
        f1 = f1_score(y_full, y_pred_full)
        
        st.markdown(f"- **Akurasi:** {accuracy:.3f}")
        st.markdown(f"- **Presisi:** {precision:.3f}")
        st.markdown(f"- **Recall:** {recall:.3f}")
        st.markdown(f"- **F1-score:** {f1:.3f}")

        st.markdown("**Confusion Matrix (pada keseluruhan data):**")
        cm = confusion_matrix(y_full, y_pred_full)
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
    on_tab_change("Segmentasi")
    st.header("Segmentasi Pasien & Karakteristik (K-Means Clustering)")
    
    cluster_labels = kmeans.labels_
    df_processed['Cluster'] = cluster_labels

    st.subheader("A. Visualisasi Segmen Pasien (PCA)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full_scaled)
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    ax_pca.set_xlabel('Principal Component 1')
    ax_pca.set_ylabel('Principal Component 2')
    ax_pca.set_title('Segmentasi Pasien Pima Indian (Visualisasi PCA)')
    legend1 = ax_pca.legend(*scatter.legend_elements(), title="Clusters")
    ax_pca.add_artist(legend1)
    st.pyplot(fig_pca)

    st.subheader("B. Analisis Data & Fitur per Segmen")
    st.markdown("**Karakteristik Rata-Rata per Segmen:**")
    cluster_stats = df_processed.groupby('Cluster')[feature_columns].mean()
    st.dataframe(cluster_stats)

    st.markdown("**Visualisasi Karakteristik Rata-Rata per Segmen:**")
    fig_bar_stats, ax_bar_stats = plt.subplots(figsize=(12, 7))
    cluster_stats.T.plot(kind='bar', ax=ax_bar_stats, colormap="viridis")
    ax_bar_stats.set_title('Perbandingan Karakteristik Rata-rata antar Segmen')
    ax_bar_stats.set_ylabel('Nilai Rata-rata Fitur')
    ax_bar_stats.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar_stats)

    with st.expander("Interpretasi Segmen & Info Model K-Means (Model Info)"):
        st.markdown("**Model:** K-Means Clustering (dengan 2 segmen)")
        
        sil_score = silhouette_score(X_full_scaled, cluster_labels)
        st.markdown(f"- **Silhouette Score:** {sil_score:.3f} (Semakin mendekati 1, semakin baik pemisahannya)")
        
        st.markdown("**Interpretasi Segmen (Contoh - Sesuaikan dengan Analisis Anda):**")
        st.markdown("""
        - **Segmen 0:** Cenderung memiliki [karakteristik dominan dari analisis Anda, misal: usia lebih muda, glukosa lebih rendah, dll.].
        - **Segmen 1:** Cenderung memiliki [karakteristik dominan dari analisis Anda, misal: usia lebih tua, BMI lebih tinggi, riwayat kehamilan lebih banyak, dll.].
        *Silakan sesuaikan deskripsi ini berdasarkan hasil analisis mendalam dari notebook atau laporan Anda.*
        """)

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
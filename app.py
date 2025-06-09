import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# HARUS DI PALING ATAS
st.set_page_config(page_title="Prediksi Penyakit Sapi", layout="centered")

# ====================
# LOAD DATA & MODEL
# ====================

# Data asli (digunakan untuk lookup penanganan & risiko)
@st.cache_data
def load_data():
    df = pd.read_csv("Diagnosa.csv")
    df = df.dropna(subset=["Gejala", "Penyakit", "Penanganan"])
    return df

df = load_data()

# Daftar semua gejala unik
def get_unique_symptoms(df):
    symptoms = df["Gejala"].str.split(",").explode().str.strip().unique()
    return sorted(symptoms)

all_symptoms = get_unique_symptoms(df)

# Buat model ML
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Gejala"])
y = df["Penyakit"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ====================
# STREAMLIT APP
# ====================

st.title("🐮 Prediksi Penyakit Sapi Berdasarkan Gejala")
st.write("Silakan pilih gejala yang dialami oleh sapi:")

# Checkbox untuk gejala
selected_symptoms = st.multiselect("Pilih Gejala:", all_symptoms)

if st.button("Prediksi Penyakit"):
    if not selected_symptoms:
        st.warning("⚠️ Pilih minimal satu gejala untuk memulai prediksi.")
    else:
        # Gabungkan input gejala menjadi string
        input_text = ", ".join(selected_symptoms)

        # Transformasi input dan prediksi
        input_vect = vectorizer.transform([input_text])
        prediction = model.predict(input_vect)[0]

        # Ambil data tambahan
        hasil = df[df["Penyakit"] == prediction].iloc[0]
        penanganan = hasil["Penanganan"]
        risiko = hasil["Risiko"] if pd.notna(hasil["Risiko"]) else "Tidak tersedia"

        # Tampilkan hasil
        st.success(f"🩺 **Hasil Prediksi Penyakit**: `{prediction}`")
        st.info(f"💊 **Penanganan yang Disarankan**:\n{penanganan}")
        st.warning(f"⚠️ **Risiko**: {risiko}")

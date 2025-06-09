import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Mengatur konfigurasi halaman Streamlit.
st.set_page_config(page_title="Prediksi Penyakit Sapi", layout="centered")
# page_title: Menentukan judul tab browser atau judul jendela aplikasi.
# layout: Mengatur tata letak halaman, 'centered' akan menempatkan konten di tengah.

# ====================
# LOAD DATA & MODEL
# ====================

# Data asli (digunakan untuk lookup penanganan & risiko)
# Menggunakan dekorator st.cache_data untuk menyimpan (cache) hasil fungsi.
# Ini berarti bahwa data hanya akan dimuat sekali ketika aplikasi dijalankan pertama kali,
# atau ketika kode fungsi ini berubah, mempercepat kinerja aplikasi.
@st.cache_data
def load_data():
    # Membaca file CSV bernama "Diagnosa.csv" ke dalam DataFrame pandas.
    df = pd.read_csv("Diagnosa.csv")
    # Menghapus baris yang memiliki nilai NaN (Not a Number) pada kolom "Gejala", "Penyakit", atau "Penanganan".
    # Memastikan data yang digunakan untuk pelatihan model dan lookup lengkap.
    df = df.dropna(subset=["Gejala", "Penyakit", "Penanganan"])
    return df

# Memanggil fungsi load_data untuk memuat data dan menyimpannya dalam variabel df.
df = load_data()

# Fungsi untuk mendapatkan daftar unik dari semua gejala yang ada di dataset.
def get_unique_symptoms(df):
    # Mengambil kolom "Gejala", memisahkannya berdasarkan koma (","),
    # lalu "meledakkan" (explode) setiap elemen menjadi baris terpisah.
    # Setelah itu, menghilangkan spasi ekstra di awal/akhir setiap gejala (strip())
    # dan mengambil nilai uniknya. Terakhir, mengurutkan gejala secara alfabetis.
    symptoms = df["Gejala"].str.split(",").explode().str.strip().unique()
    return sorted(symptoms)

# Memanggil fungsi get_unique_symptoms untuk mendapatkan daftar semua gejala unik.
all_symptoms = get_unique_symptoms(df)

# Buat model ML
# Menginisialisasi CountVectorizer. Objek ini akan digunakan untuk mengubah teks (gejala)
# menjadi representasi numerik yang dapat dipahami oleh model Machine Learning.
vectorizer = CountVectorizer()
# Melatih CountVectorizer pada kolom "Gejala" dari DataFrame df dan mengubahnya menjadi matriks fitur X.
# X adalah matriks yang menunjukkan frekuensi kemunculan setiap gejala dalam setiap baris data.
X = vectorizer.fit_transform(df["Gejala"])
# Mengambil kolom "Penyakit" sebagai variabel target (y) untuk model klasifikasi.
y = df["Penyakit"]

# Menginisialisasi model Random Forest Classifier.
# n_estimators=100: Menentukan jumlah pohon keputusan dalam forest (100 pohon).
# random_state=42: Mengatur seed untuk reproduksibilitas hasil.
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Melatih model Random Forest menggunakan data fitur X dan label target y.
model.fit(X, y)

# ====================
# STREAMLIT APP
# ====================

# Menampilkan judul utama aplikasi di Streamlit.
st.title("🐮 Prediksi Penyakit Sapi Berdasarkan Gejala")
# Menampilkan teks instruksi di bawah judul.
st.write("Silakan pilih gejala yang dialami oleh sapi:")

# Membuat widget multiselect di Streamlit.
# Pengguna dapat memilih beberapa gejala dari daftar all_symptoms.
# Pilihan yang dipilih akan disimpan dalam variabel selected_symptoms.
selected_symptoms = st.multiselect("Pilih Gejala:", all_symptoms)

# Membuat tombol di Streamlit dengan label "Prediksi Penyakit".
# Kode di dalam blok if ini akan dieksekusi ketika tombol diklik.
if st.button("Prediksi Penyakit"):
    # Memeriksa apakah pengguna telah memilih gejala.
    if not selected_symptoms:
        # Jika tidak ada gejala yang dipilih, tampilkan pesan peringatan.
        st.warning("⚠️ Pilih minimal satu gejala untuk memulai prediksi.")
    else:
        # Jika ada gejala yang dipilih, gabungkan gejala-gejala tersebut menjadi satu string,
        # dipisahkan oleh koma dan spasi.
        input_text = ", ".join(selected_symptoms)

        # Transformasi input dan prediksi
        # Mengubah string gejala yang dipilih pengguna menjadi representasi numerik
        # menggunakan vectorizer yang sudah dilatih.
        input_vect = vectorizer.transform([input_text])
        # Melakukan prediksi menggunakan model yang sudah dilatih.
        # [0] digunakan untuk mengambil hasil prediksi pertama (karena predict mengembalikan array).
        prediction = model.predict(input_vect)[0]

        # Ambil data tambahan
        # Mencari baris di DataFrame df yang memiliki nilai "Penyakit" sama dengan hasil prediksi.
        # iloc[0] mengambil baris pertama jika ada beberapa baris yang cocok (asumsi penyakit unik).
        hasil = df[df["Penyakit"] == prediction].iloc[0]
        # Mengambil nilai kolom "Penanganan" dari hasil yang ditemukan.
        penanganan = hasil["Penanganan"]
        # Mengambil nilai kolom "Risiko" dari hasil yang ditemukan.
        # Memeriksa apakah nilai risiko adalah NaN (menggunakan pd.notna).
        # Jika NaN, set risiko menjadi "Tidak tersedia".
        risiko = hasil["Risiko"] if pd.notna(hasil["Risiko"]) else "Tidak tersedia"

        # Tampilkan hasil
        # Menampilkan hasil prediksi penyakit dengan format sukses (warna hijau).
        st.success(f"🩺 **Hasil Prediksi Penyakit**: `{prediction}`")
        # Menampilkan penanganan yang disarankan dengan format informasi (warna biru).
        st.info(f"💊 **Penanganan yang Disarankan**:\n{penanganan}")
        # Menampilkan informasi risiko dengan format peringatan (warna kuning).
        st.warning(f"⚠️ **Risiko**: {risiko}")
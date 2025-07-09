import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import numpy as np
import difflib
import re
import os

# ======================= FUNGSI UTAMA =======================

@st.cache_resource
def muat_model(path_model, path_encoder):
    try:
        model = pickle.load(open(path_model, "rb"))
        encoder = pickle.load(open(path_encoder, "rb"))
        return model, encoder
    except Exception as e:
        st.error(f"❌ Gagal memuat model/encoder: {e}")
        st.stop()

@st.cache_data
def muat_data(path_csv):
    try:
        df = pd.read_csv(path_csv)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.stop()

def bersihkan_nama_jurnal(nama):
    nama = re.sub(r"\s+", " ", str(nama).strip())
    nama = re.sub(r"[^A-Za-z0-9\s]", "", nama)
    return nama

def buat_singkatan_canggih(nama):
    """Membuat singkatan jurnal yang lebih pintar."""
    stopwords = {"of", "in", "and", "the"}
    kata = nama.split()
    huruf = []
    for w in kata:
        if w.lower() in stopwords:
            continue
        if len(w) <= 3:
            huruf.append(w[:2].upper())
        else:
            huruf.append(w[0].upper())
    return "".join(huruf)

# ======================= KONFIGURASI =======================

MODEL_PATH = "quartil_model_all.pkl"
ENCODER_PATH = "encoder_all.pkl"
CSV_PATH = "jurnal_dataset_final.csv"
LOGO_PATH = "logo_scopusquartile_s.svg"

# ======================= LOAD DATA & MODEL =======================

model, encoder = muat_model(MODEL_PATH, ENCODER_PATH)
df = muat_data(CSV_PATH)

# ======================= PREPROCESSING DATA =======================

df["Nama Jurnal"] = df["Nama Jurnal"].apply(bersihkan_nama_jurnal)
df["Bidang Ilmu"] = df["Bidang Ilmu"].str.title()
df["ISSN"] = df["ISSN"].astype(str).str.replace(" ", "")
df["Abbreviation"] = df["Nama Jurnal"].apply(buat_singkatan_canggih)
df["Nama Lower"] = df["Nama Jurnal"].str.lower()

# ======================= KONFIGURASI HALAMAN =======================

st.set_page_config(
    page_title="ScopusQuartile AI",
    page_icon="📈",
    layout="centered"
)

# ======================= LOGO =======================
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "r") as f:
        svg_logo = f.read()
    st.markdown(f"<div style='text-align:center;'>{svg_logo}</div>", unsafe_allow_html=True)

# ======================= TAGLINE (opsional) =======================
# st.write("*Prediksi Quartil Jurnal dengan AI*")

# ======================= PILIHAN JURNAL (AUTO-COMPLETE) =======================
list_pilihan = df.apply(
    lambda row: f"{row['Nama Jurnal']} [{row['Abbreviation']}]",
    axis=1
).tolist()

nama_jurnal_pilih = st.selectbox(
    "🔍 Pilih jurnal atau cari berdasarkan nama/singkatan",
    options=[""] + sorted(list_pilihan),
    index=0
)

# ======================= LOGIKA UTAMA =======================
if nama_jurnal_pilih != "":
    nama_jurnal = nama_jurnal_pilih.split(" [")[0]
    nama_input = nama_jurnal.strip().lower()

    if nama_input in df["Nama Lower"].values:
        jurnal = df[df["Nama Lower"] == nama_input].iloc[0]

        st.success("✅ Jurnal Ditemukan")
        st.markdown(f"""
        **Nama Jurnal:** {jurnal['Nama Jurnal']}  
        **Singkatan Jurnal:** {jurnal['Abbreviation']}  
        **ISSN:** {jurnal['ISSN']}  
        **SJR:** {jurnal['SJR']}  
        **H-Index:** {jurnal['H-Index']}  
        **Bidang Ilmu:** {jurnal['Bidang Ilmu']}
        """)

        # Siapkan fitur input model
        X_num = np.array([[jurnal["SJR"], jurnal["H-Index"]]])
        try:
            X_cat = encoder.transform([[jurnal["Bidang Ilmu"]]])
        except ValueError:
            X_cat = np.zeros((1, encoder.categories_[0].shape[0]))

        X_input = np.hstack((X_num, X_cat))

        # Prediksi
        probabilities = model.predict_proba(X_input)[0]
        classes = model.classes_
        pred_idx = np.argmax(probabilities)
        pred_quartil = classes[pred_idx]
        confidence = probabilities[pred_idx]

        st.subheader


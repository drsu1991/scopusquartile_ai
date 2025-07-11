import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import numpy as np
import difflib
import re

# 🎯 Load model dan encoder
model = pickle.load(open("quartil_model_all.pkl", "rb"))
encoder = pickle.load(open("encoder_all.pkl", "rb"))

# 🎯 Load dataset
df = pd.read_csv("jurnal_dataset_final.csv")

# 🎯 Cleaning tambahan
df["Nama Jurnal"] = df["Nama Jurnal"].apply(lambda x: re.sub(r"\s+", " ", str(x).strip()))
df["Nama Jurnal"] = df["Nama Jurnal"].apply(lambda x: re.sub(r"[^A-Za-z0-9\s]", "", x))
df["Bidang Ilmu"] = df["Bidang Ilmu"].str.title()
df["ISSN"] = df["ISSN"].astype(str).str.replace(" ", "")

# 🎯 Buat singkatan canggih
def buat_singkatan_canggih(nama):
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

df["Abbreviation"] = df["Nama Jurnal"].apply(buat_singkatan_canggih)

# 🎯 Kolom bantu lowercase
df["Nama Lower"] = df["Nama Jurnal"].str.lower()

# 🎯 Setup halaman (icon 📈)
st.set_page_config(
    page_title="ScopusQuartile AI",
    page_icon="📈",
    layout="centered"
)

# 🎯 Tampilkan logo SVG
with open("logo_scopusquartile_s.svg", "r") as f:
    svg_logo = f.read()

st.markdown(
    f"<div style='text-align:center;'>{svg_logo}</div>",
    unsafe_allow_html=True
)

# 🎯 1 tagline di bawah logo
# (Jika SVG sudah ada tulisan/tagline, baris ini boleh dihapus)
# st.write("*AI-Powered Quartile Prediction*")

# 🎯 Auto-complete input (default kosong)
list_pilihan = df.apply(
    lambda row: f"{row['Nama Jurnal']} [{row['Abbreviation']}]",
    axis=1
).tolist()

nama_jurnal_pilih = st.selectbox(
    "🔍 Pilih jurnal atau cari berdasarkan nama/singkatan",
    options=[""] + sorted(list_pilihan),
    index=0
)

if nama_jurnal_pilih != "":
    # Ambil nama asli
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

        # 🎯 Siapkan input fitur
        X_num = np.array([[jurnal["SJR"], jurnal["H-Index"]]])
        try:
            X_cat = encoder.transform([[jurnal["Bidang Ilmu"]]])
        except ValueError:
            X_cat = np.zeros((1, encoder.categories_[0].shape[0]))

        X_input = np.hstack((X_num, X_cat))

        # 🎯 Prediksi
        probabilities = model.predict_proba(X_input)[0]
        classes = model.classes_
        pred_idx = np.argmax(probabilities)
        pred_quartil = classes[pred_idx]
        confidence = probabilities[pred_idx]

        st.subheader(f"🎯 Prediksi Quartil: {pred_quartil}")
        st.write(f"Confidence: **{confidence*100:.1f}%**")

        # 🎯 Download hasil PDF
        if st.button("⬇️ Download Hasil PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200,10,txt="Hasil Prediksi Quartil Jurnal", ln=1, align="C")
            pdf.cell(200,10,txt=f"Nama Jurnal: {jurnal['Nama Jurnal']}", ln=2)
            pdf.cell(200,10,txt=f"Singkatan: {jurnal['Abbreviation']}", ln=3)
            pdf.cell(200,10,txt=f"ISSN: {jurnal['ISSN']}", ln=4)
            pdf.cell(200,10,txt=f"SJR: {jurnal['SJR']}", ln=5)
            pdf.cell(200,10,txt=f"H-Index: {jurnal['H-Index']}", ln=6)
            pdf.cell(200,10,txt=f"Bidang Ilmu: {jurnal['Bidang Ilmu']}", ln=7)
            pdf.cell(200,10,txt=f"Prediksi Quartil: {pred_quartil}", ln=8)
            pdf.cell(200,10,txt=f"Confidence: {confidence*100:.1f}%", ln=9)
            pdf.output("hasil_prediksi.pdf")

            with open("hasil_prediksi.pdf", "rb") as file:
                st.download_button(
                    label="Download PDF",
                    data=file,
                    file_name="hasil_prediksi.pdf",
                    mime="application/pdf"
                )

    else:
        st.error("❌ Nama jurnal tidak ditemukan.")
        daftar_jurnal = df["Nama Lower"].tolist()
        rekomendasi = difflib.get_close_matches(nama_input, daftar_jurnal, n=5, cutoff=0.5)
        if rekomendasi:
            st.info("🔍 Apakah maksud Anda salah satu dari berikut?")
            for r in rekomendasi:
                nama_asli = df[df["Nama Lower"] == r]["Nama Jurnal"].values[0]
                st.write(f"- {nama_asli}")
        else:
            st.warning("Tidak ada rekomendasi nama jurnal mirip ditemukan.")

# 🎯 Footer
st.markdown("---")
st.markdown("""
👤 **Pengembang Aplikasi**  
**dr. Suhendra Mandala Ernas**  
PPDS Patologi Klinik FK Unair – RSUD dr. Soetomo  

🔗 [ORCID](https://orcid.org/0009-0007-1290-1673)  
🔗 [LinkedIn](https://www.linkedin.com/in/drsuhendramandalaernas/)  

💼 *Powered by ScopusQuartile AI*
""")


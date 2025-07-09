import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
# Load Data & Model
# ============================
@st.cache_data
def load_data():
    return pd.read_csv("jurnal_dataset_final.csv")

@st.cache_resource
def load_model():
    with open("quartil_model_all.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder_all.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

df = load_data()
model, encoder = load_model()

# ============================
# Config Page
# ============================
st.set_page_config(
    page_title="ScopusQuartile AI",
    page_icon="📈",
    layout="centered"
)

# ============================
# Header
# ============================
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("logo_scopusquartile_s.svg", width=50)
with col2:
    st.markdown("## **ScopusQuartile AI**\n_AI-Powered Quartile Prediction_")

st.markdown("---")

# ============================
# Input Search
# ============================
query = st.text_input("🔍 Ketik nama jurnal atau singkatannya")

# ============================
# Search Logic
# ============================
if query:
    matches = df[
        df["Nama Jurnal"].str.contains(query, case=False, na=False) |
        df["Singkatan"].str.contains(query, case=False, na=False)
    ]

    if not matches.empty:
        pilihan = st.selectbox(
            "Pilih jurnal yang sesuai:",
            options=matches.index,
            format_func=lambda i: f"{matches.loc[i,'Nama Jurnal']} [{matches.loc[i,'Singkatan']}]"
        )
        jurnal = matches.loc[pilihan]

        st.success("✅ Profil Jurnal Ditemukan")

        # Info Jurnal
        st.markdown(f"**Nama Jurnal:** {jurnal['Nama Jurnal']}")
        st.markdown(f"**Singkatan Jurnal:** {jurnal['Singkatan']}")
        st.markdown(f"**ISSN:** {jurnal['ISSN']}")
        st.markdown(f"**SJR:** {jurnal['SJR']}")
        st.markdown(f"**H-Index:** {jurnal['H-Index']}")
        st.markdown(f"**Bidang Ilmu:** {jurnal['Bidang Ilmu']}")

        # Prediksi Quartile
        X_cat = encoder.transform([[jurnal["Bidang Ilmu"]]])
        X_num = [[jurnal["SJR"], jurnal["H-Index"]]]
        X_input = np.hstack([X_num, X_cat])
        quartile_pred = model.predict(X_input)[0]

        st.markdown("---")
        st.markdown(f"🎯 **Prediksi Quartile: `{quartile_pred}`**")
    else:
        st.warning("⚠️ Jurnal tidak ditemukan. Cek ejaan atau coba kata kunci lain.")
else:
    st.info("💡 Masukkan kata kunci di atas untuk memulai pencarian.")

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("Developed by dr. Suhendra Mandala Ernas | [ORCID](https://orcid.org/0009-0007-1290-1673) | [LinkedIn](https://www.linkedin.com/in/drsuhendramandalaernas/)")

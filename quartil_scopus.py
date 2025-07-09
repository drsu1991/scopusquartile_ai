import streamlit as st
import pandas as pd
import pickle

# ===========================
# Load data dan model
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("jurnal_dataset_final.csv")
    return df

@st.cache_resource
def load_model():
    with open("quartil_model_all.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder_all.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

df = load_data()
model, encoder = load_model()

# ===========================
# Judul dan logo
# ===========================
st.set_page_config(
    page_title="ScopusQuartile AI",
    page_icon="📈",
    layout="centered"
)

col1, col2 = st.columns([0.1,0.9])
with col1:
    st.image("logo_scopusquartile_s.svg", width=50)
with col2:
    st.markdown("## **ScopusQuartile AI**\n_AI-Powered Quartile Prediction_")

# ===========================
# Input pencarian
# ===========================
query = st.text_input("🔍 Ketik nama jurnal atau singkatannya")

# ===========================
# Tampilkan hasil filter
# ===========================
if query:
    matches = df[
        df["Nama Jurnal"].str.contains(query, case=False, na=False) |
        df["Singkatan"].str.contains(query, case=False, na=False)
    ]

    if not matches.empty:
        # Pilih jurnal dari hasil pencarian
        pilihan = st.selectbox(
            "Pilih jurnal:",
            matches.index,
            format_func=lambda i: f"{matches.loc[i,'Nama Jurnal']} [{matches.loc[i,'Singkatan']}]"
        )
        
        jurnal = matches.loc[pilihan]
        st.success("✅ Profil Jurnal Ditemukan")

        st.markdown(f"**Nama Jurnal:** {jurnal['Nama Jurnal']}")
        st.markdown(f"**Singkatan:** {jurnal['Singkatan']}")
        st.markdown(f"**ISSN:** {jurnal['ISSN']}")
        st.markdown(f"**SJR:** {jurnal['SJR']}")
        st.markdown(f"**H-Index:** {jurnal['H-Index']}")
        st.markdown(f"**Bidang Ilmu:** {jurnal['Bidang Ilmu']}")

        # Prediksi Quartile
        X_cat = encoder.transform([[jurnal["Bidang Ilmu"]]])
        X_num = [[jurnal["SJR"], jurnal["H-Index"]]]
        X_input = pd.DataFrame(
            np.hstack([X_num, X_cat]),
            columns=["SJR", "H-Index"] + list(encoder.get_feature_names_out())
        )
        quartil_pred = model.predict(X_input)[0]
        st.markdown(f"🎯 **Prediksi Quartile: {quartil_pred}**")
    else:
        st.warning("⚠️ Jurnal tidak ditemukan. Cek ejaan atau coba kata kunci lain.")
else:
    st.info("Masukkan kata kunci jurnal di atas untuk memulai pencarian.")

# Footer
st.markdown("---")
st.caption("Developed by dr. Suhendra Mandala Ernas | [ORCID](https://orcid.org/0009-0007-1290-1673) | [LinkedIn](https://www.linkedin.com/in/drsuhendramandalaernas/)")

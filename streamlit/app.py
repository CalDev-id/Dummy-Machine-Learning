import streamlit as st
from utils.inference import preprocess_image, predict_image
from PIL import Image

st.set_page_config(page_title="🖼️ Calorify CNN Classifier", layout="centered")
st.title("🍽️ Calorify Image Classification (CNN)")
st.write("Upload gambar makanan/minuman, dan model akan mengklasifikasikannya berdasarkan kadar kalori.")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Prediksi"):
        img_array = preprocess_image(uploaded_file)
        label, confidence = predict_image(img_array)
        st.success(f"Prediksi: **{label}** (Probabilitas: {confidence:.2f})")

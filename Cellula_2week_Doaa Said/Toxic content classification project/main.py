import streamlit as st
from db import init_db, add_entry, get_all_entries
from infer import classify_text
from imagecaption import generate_caption
import tempfile

st.title("🧹 Toxic Content Classifier")

init_db()

menu = st.sidebar.selectbox("Menu", ["Classify Text", "Classify Image", "View Database"])

if menu == "Classify Text":
    user_input = st.text_area("Enter text to classify")
    if st.button("Classify Text"):
        result = classify_text(user_input)
        add_entry(user_input, result)
        st.success(f"Classification Result: {result}")

elif menu == "Classify Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_image.getbuffer())
            img_path = tmp.name
        caption = generate_caption(img_path)
        st.image(uploaded_image, caption=f"Generated Caption: {caption}")
        if st.button("Classify Image Caption"):
            result = classify_text(caption)
            add_entry(caption, result)
            st.success(f"Classification Result: {result}")

elif menu == "View Database":
    st.subheader("Stored Inputs and Classifications")
    st.dataframe(get_all_entries())

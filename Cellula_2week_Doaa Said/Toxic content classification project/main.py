import streamlit as st
import infer
import imagecaption, db
from PIL import Image
import pandas as pd
import io

st.title("Toxicity Classification")
db.init_db()

menu = st.sidebar.radio(
    "Choose Task",
    [
        "Classification",
        "Show Database"
    ]
)

# --- CLASSIFICATION 
if menu == "Classification":
    query = st.text_area("Enter query text (optional):")
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

    img_desc = ""
    if uploaded_file:
        # Reset pointer and open with PIL
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        #  Reset pointer again before passing to BLIP
        uploaded_file.seek(0)
        img_desc = imagecaption.generate_caption(uploaded_file)
        st.info(f"BLIP Caption: {img_desc}")

    if st.button("Classify"):
        label = infer.classify(query, img_desc)
        st.success(f"Prediction: {label}")

        # Save to DB (store text, caption, or both)
        combined_input = query if query else ""
        if img_desc:
            combined_input = (combined_input + " | " if combined_input else "") + img_desc
        db.insert_record(combined_input, label)

# --- DATABASE VIEW ---
elif menu == "Show Database":
    st.subheader("📊 Stored Classification Results")
    records = db.fetch_all()
    if records:
        df = pd.DataFrame(records, columns=["ID", "Query", "Label", "Timestamp"])
        st.dataframe(df)
    else:
        st.info("No records found in the database.")


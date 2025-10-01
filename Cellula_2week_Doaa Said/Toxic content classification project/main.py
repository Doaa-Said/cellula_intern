import streamlit as st
import infer
import imagecaption, db
from PIL import Image

st.title("Toxicity classification")
db.init_db()
menu = st.sidebar.radio(
    "Choose Task",
    [
        "Text Classification",
        "Image Text Classification",
        "Show Database"
    ]
)

if menu == "Text Classification":
    query = st.text_area("Enter query text:")
    img_desc = st.text_area("Enter image description:")
    if st.button("Classify"):
        label = infer.classify(query, img_desc)
        st.success(f"Prediction: {label}")
        db.insert_record(query, label)



elif menu == "Image Text Classification":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Image"):
            # Use BLIP captioning only
            img_desc = imagecaption.generate_caption(uploaded_file)

            # Classify based on BLIP caption
            label = infer.classify(img_desc)

            st.success(f"Prediction: {label} (BLIP Caption Only)")

            # Save to DB
            db.insert_record(img_desc, label)
elif menu == "Show Database":
    st.subheader("📊 Stored Classification Results")
    records = db.fetch_all()
    if records:
        import pandas as pd
        df = pd.DataFrame(records, columns=["ID", "Query",  "Label", "Timestamp"])
        st.dataframe(df)
    else:
        st.info("No records found in the database.")

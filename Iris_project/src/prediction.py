# src/prediction.py
import streamlit as st
import os

def predict_species(lr_model, le, species_image_dirs):
    st.subheader("🔮 Predict Iris Species")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sw = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)

    with col2:
        pl = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
        pw = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

    if st.button("Predict"):
        sample = [[sl, sw, pl, pw]]
        prediction = lr_model.predict(sample)
        species = le.inverse_transform(prediction)[0]

        st.markdown(f"<h3 style='color:green'> Predicted Species: <b>{species}</b></h3>", unsafe_allow_html=True)

        # Load multiple images
        image_dir = species_image_dirs.get(species)

        if image_dir and os.path.exists(image_dir):
            images = [
                os.path.join(image_dir, img)
                for img in os.listdir(image_dir)
                if img.lower().endswith((".jpg", ".png", ".jpeg",".webp"))
            ]

            st.markdown("### 🌸 Sample Images")

            cols = st.columns(min(3, len(images)))
            for col, img_path in zip(cols, images):
                col.image(img_path, use_container_width=True)

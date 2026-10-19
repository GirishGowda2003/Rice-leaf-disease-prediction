import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Rice Leaf Disease Prediction",
    page_icon="🌾",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "Rice_leaf_transfer_model.h5",
        compile=False
    )

# ---------------- CLASS NAMES ----------------
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# ---------------- DASHBOARD HEADER ----------------
st.title("🌾 Rice Leaf Disease Prediction")
st.write(
    "Upload a rice leaf image to predict the disease "
    "using a trained MobileNetV2 deep learning model."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("About the Project")

st.sidebar.write(
    """
    **Model:** MobileNetV2

    **Task:** Image Classification

    **Input:** Rice Leaf Image

    **Output:** Predicted Disease
    """
)

st.sidebar.info(
    "Upload a clear rice leaf image for prediction."
)

# ---------------- LOAD MODEL ----------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Rice Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Rice Leaf",
            use_container_width=True
        )


    st.subheader("Prediction Results")

    if uploaded_file is not None:

        if st.button("🔍 Predict Disease", type="primary"):

            with st.spinner("Analyzing image..."):

                # Preprocess image
                processed_image = preprocess_image(image)

                # Model prediction
                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                # Validate output
                if len(predictions) != len(class_names):

                    st.error(
                        f"Model returned {len(predictions)} "
                        f"outputs, but {len(class_names)} "
                        "classes are configured."
                    )

                else:
                    # Get predicted class
                    predicted_index = np.argmax(predictions)

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = predictions[
                        predicted_index
                    ] * 100

                    # Display results
                    st.success(
                        f"Predicted Disease: {predicted_class}"
                    )

                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.2f}%"
                    )

                    # Probability table
                    st.subheader("Class Probabilities")

                    results = pd.DataFrame({
                        "Disease": class_names,
                        "Probability (%)": predictions * 100
                    })

                    results = results.sort_values(
                        by="Probability (%)",
                        ascending=False
                    )

                    st.bar_chart(
                        results.set_index("Disease")
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )

    else:
        st.info(
            "Upload an image to see the prediction results."
        )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Rice Leaf Disease Prediction | "
    "Deep Learning with MobileNetV2"
)
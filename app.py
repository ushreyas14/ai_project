import streamlit as st
import numpy as np
import json
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Dermatology Classifier", layout="centered", page_icon="🧬")

st.title("Dermatology Image Classifier")
st.write("Upload a skin lesion image to get an AI diagnostic prediction.")

MODEL_PATH = 'dermatology_model.h5'
STATS_PATH = 'normalization_stats.json'

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        return None, None
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        return model, stats
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, stats = load_assets()

if model is None:
    st.warning("⚠️ **Model files not found!**\n\nYou need to run your Jupyter Notebook fully to the end to save `dermatology_model.h5` and `normalization_stats.json` before this UI will function.")
else:
    # pd.Categorical codes are assigned alphabetically
    lesion_type_dict = {
        0: 'Actinic keratoses (akiec)',
        1: 'Basal cell carcinoma (bcc)',
        2: 'Benign keratosis-like lesions (bkl)',
        3: 'Dermatofibroma (df)',
        4: 'Melanoma (mel)',
        5: 'Melanocytic nevi (nv)',
        6: 'Vascular lesions (vasc)'
    }

    uploaded_file = st.file_uploader("Choose a dermatology image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
        with st.spinner("Processing..."):
            # Preprocess to match training shape precisely
            img = image.resize((32, 32))
            img_array = np.array(img).astype(np.float32)
            
            # Apply identical z-normalization generated during training
            mean = stats['mean']
            std = stats['std']
            img_array = (img_array - mean) / (std + 1e-8)
            
            # Expand dims to represent batch size of 1
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0]
            max_idx = int(np.argmax(pred))
            confidence = float(pred[max_idx])

            diagnosis = lesion_type_dict.get(max_idx, "Unknown")
            
            with col2:
                st.subheader("AI Prediction")
                if confidence > 0.8:
                    st.success(f"**{diagnosis}**")
                elif confidence > 0.5:
                    st.warning(f"**{diagnosis}**")
                else:
                    st.error(f"**{diagnosis}**")
                    
                st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%")
                
                with st.expander("Show probabilities"):
                    for i, prob in enumerate(pred):
                        st.write(f"{lesion_type_dict[i]}: {prob*100:.2f}%")

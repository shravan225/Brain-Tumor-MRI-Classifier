import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import os

# Load Model
@st.cache_resource
def load_tumor_model():
    return load_model('brain_tumor_resnet.h5')

model = load_tumor_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# App UI
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("""
Upload an MRI scan to classify between:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor
""")

# File Upload
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    st.image(uploaded_file, caption="Uploaded MRI", width=300)
    
    # Preprocess
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    
    # Predict with progress
    with st.spinner('Analyzing MRI...'):
        time.sleep(1)  # Simulate processing
        pred = model.predict(x)[0]
    
    # Results
    st.success("Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", class_names[np.argmax(pred)])
        st.metric("Confidence", f"{np.max(pred)*100:.2f}%")
    
    with col2:
        # Confidence bars
        st.write("Class Probabilities:")
        for i, (cls, prob) in enumerate(zip(class_names, pred)):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
    
    # Interpretation
    st.subheader("Clinical Note:")
    if np.argmax(pred) == 2:  # notumor
        st.info("No tumor detected. Recommend routine follow-up.")
    else:
        st.warning(f"Potential {class_names[np.argmax(pred)]} detected. Urgent specialist review recommended.")
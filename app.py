import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

treatment_suggestions = {
    "cataract": """
    **Suggested Next Steps:**
    • Consult an ophthalmologist for slit-lamp examination  
    • Vision correction (glasses) may help in early stages  
    • Cataract surgery may be recommended if vision is significantly affected
    """,

    "diabetic_retinopathy": """
    **Suggested Next Steps:**
    • Immediate consultation with a retina specialist  
    • Maintain strict blood sugar control  
    • Fundus fluorescein angiography may be advised  
    • Treatments may include laser therapy or anti-VEGF injections
    """,

    "glaucoma": """
    **Suggested Next Steps:**
    • Urgent ophthalmology evaluation  
    • Measure intraocular pressure (IOP)  
    • Medicated eye drops may be prescribed  
    • Regular monitoring to prevent vision loss
    """,

    "normal": """
    **Suggested Next Steps:**
    • No signs of major eye disease detected  
    • Maintain regular eye check-ups (once every 1–2 years)  
    • Follow a healthy lifestyle and protect eyes from strain
    """
}


model = tf.keras.models.load_model(
    "C:\\Users\\aakan\\Downloads\\best_model_val_acc.h5"
)

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

st.title("Eye Disease Classification")

uploaded_file = st.file_uploader(
    "Upload fundus image", type=["jpg","png","jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=650)

    img = image.resize((224, 224))
    img = np.array(img)

    
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    predicted_label = class_names[class_id]

    st.subheader(f"Prediction: {class_names[class_id]}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    st.write("Raw probabilities:", prediction)

    st.subheader("Suggested Treatment / Next Steps")
    st.info(treatment_suggestions[predicted_label])

    st.warning(
    "Disclaimer: This application is for educational and screening purposes only. "
    "It is not a medical diagnosis. Please consult a qualified ophthalmologist."
    )

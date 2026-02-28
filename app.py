import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import InferenceClient
import requests

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Optional for Places API

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL_PATH = "best_model_val_acc.h5"   
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Main header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-warning {
    border-left: 3px solid var(--warning);
}
.card-danger {
    border-left: 3px solid var(--danger);
}
.card-success {
    border-left: 3px solid var(--success);
}

/* Prediction badge */
.pred-badge {
    display: inline-block;
    padding: 0.5rem 1.25rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.pred-normal { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.pred-warning { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }
.pred-danger { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid #ef4444; }

/* Confidence bar */
.conf-bar-bg {
    background: var(--surface2);
    border-radius: 999px;
    height: 8px;
    margin-top: 0.5rem;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* Doctor card */
.doctor-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.doctor-card:hover { border-color: var(--accent); }
.doc-name { font-weight: 600; font-size: 1rem; color: var(--text); }
.doc-spec { color: var(--accent); font-size: 0.85rem; font-weight: 500; }
.doc-meta { color: var(--muted); font-size: 0.82rem; }

/* Chat bubble */
.chat-user {
    background: linear-gradient(135deg, var(--accent2), #5b21b6);
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    margin-left: 20%;
    color: white;
    font-size: 0.92rem;
}
.chat-bot {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    margin-right: 20%;
    font-size: 0.92rem;
    color: var(--text);
}
.chat-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
    color: var(--muted);
}

/* Step indicators */
.step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.step:last-child { border-bottom: none; }
.step-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; color: white; flex-shrink: 0;
}

/* Metrics */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--accent); }
.metric-label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* Override streamlit defaults */
.stFileUploader { border: 2px dashed var(--border) !important; border-radius: 12px !important; }
.stTextInput input, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }
div[data-testid="stAlert"] {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA ──────────────────────────────────────────────────────────────────────
CONDITION_INFO = {
    "cataract": {
        "severity": "moderate",
        "badge_class": "pred-warning",
        "icon": "🔵",
        "description": "A cataract is a clouding of the eye's natural lens. It is the leading cause of blindness worldwide but is highly treatable.",
        "steps": [
            "Schedule a consultation with an ophthalmologist within 2–4 weeks",
            "Get a slit-lamp biomicroscopy examination",
            "Ask about corrective lenses for early-stage management",
            "Discuss cataract surgery (phacoemulsification) if vision is significantly impaired",
            "Inquire about intraocular lens (IOL) options",
            "Plan follow-up every 6–12 months post-surgery"
        ],
        "specialist": "Ophthalmologist / Cataract Surgeon",
        "urgency": "Non-urgent — within 2–4 weeks",
        "lifestyle": ["Wear UV-protective sunglasses", "Maintain a diet rich in antioxidants (Vitamins C & E)", "Avoid smoking", "Control diabetes and blood pressure"]
    },
    "diabetic_retinopathy": {
        "severity": "severe",
        "badge_class": "pred-danger",
        "icon": "🔴",
        "description": "Diabetic retinopathy is a diabetes complication affecting the retinal blood vessels. It is a leading cause of vision loss in working-age adults.",
        "steps": [
            "Seek urgent retina specialist consultation within 1 week",
            "Undergo fundus fluorescein angiography (FFA)",
            "Achieve strict blood sugar control (HbA1c < 7%)",
            "Monitor blood pressure and cholesterol",
            "Discuss anti-VEGF injection therapy (Ranibizumab/Bevacizumab)",
            "Consider laser photocoagulation if indicated",
            "Vitrectomy may be required for advanced cases"
        ],
        "specialist": "Retina Specialist / Vitreoretinal Surgeon",
        "urgency": "Urgent — within 1 week",
        "lifestyle": ["Strict blood glucose control", "Regular HbA1c monitoring", "Blood pressure management", "Avoid smoking and alcohol", "Annual dilated eye exams"]
    },
    "glaucoma": {
        "severity": "severe",
        "badge_class": "pred-danger",
        "icon": "🟡",
        "description": "Glaucoma is a group of eye conditions that damage the optic nerve, often caused by elevated intraocular pressure. Vision loss is irreversible but can be halted.",
        "steps": [
            "Urgent ophthalmology evaluation — within 1 week",
            "Measure intraocular pressure (IOP tonometry)",
            "Visual field testing (perimetry)",
            "OCT (optical coherence tomography) of optic nerve",
            "Start medicated eye drops as prescribed (prostaglandins, beta-blockers)",
            "Consider laser trabeculoplasty (SLT/ALT)",
            "Surgical options: trabeculectomy or MIGS if drops are insufficient",
            "Regular monitoring every 3–6 months"
        ],
        "specialist": "Glaucoma Specialist / Ophthalmologist",
        "urgency": "Urgent — within 1 week",
        "lifestyle": ["Avoid prolonged head-down positions", "Limit caffeine intake", "Regular aerobic exercise", "Sleep with head slightly elevated", "Never skip prescribed eye drops"]
    },
    "normal": {
        "severity": "normal",
        "badge_class": "pred-normal",
        "icon": "🟢",
        "description": "No signs of significant eye disease detected in the fundus image. Continue routine eye care for long-term ocular health.",
        "steps": [
            "Schedule routine eye exam once every 1–2 years",
            "Undergo dilated fundus exam annually if diabetic or over 40",
            "Monitor for any changes in vision (blurriness, floaters, flashes)",
            "Maintain healthy lifestyle habits"
        ],
        "specialist": "General Ophthalmologist (routine checkup)",
        "urgency": "Routine — no immediate action needed",
        "lifestyle": ["Eat a balanced diet rich in leafy greens & fish", "Take regular screen breaks (20-20-20 rule)", "Wear UV-protective sunglasses outdoors", "Quit smoking if applicable", "Exercise regularly to maintain healthy blood pressure"]
    }
}

DOCTOR_SPECIALTIES = {
    "cataract": ["Cataract Surgeon", "Ophthalmologist"],
    "diabetic_retinopathy": ["Retina Specialist", "Vitreoretinal Surgeon"],
    "glaucoma": ["Glaucoma Specialist", "Ophthalmologist"],
    "normal": ["Ophthalmologist"]
}

# ─── HELPERS ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

def classify_image(model, image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return CLASS_NAMES[class_id], confidence, preds[0]

def get_hospitals_nominatim(location: str, condition: str):
    """Use OpenStreetMap Nominatim + Overpass API — free, no API key needed."""
    try:
        encoded_location = requests.utils.quote(location)
        headers = {
            "User-Agent": "RetinaScan-AI/1.0 (aakanksha21c@email.com)",  # Must be unique & real
            "Accept-Language": "en"
        }

        # Geocode location
        geo_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(location)}&format=json&limit=1"
        geo_resp = requests.get(geo_url, headers=headers, timeout=25)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return None, None
        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])
        display_name = geo_data[0]["display_name"]

        # Overpass query for hospitals/clinics
        overpass_query = f"""
        [out:json][timeout:10];
        (
          node["amenity"="hospital"](around:10000,{lat},{lon});
          node["amenity"="clinic"](around:10000,{lat},{lon});
          node["healthcare"="hospital"](around:10000,{lat},{lon});
        );
        out body 6;
        """
        ov_resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            headers={"User-Agent": "RetinaScan-AI/1.0 (aakanksha21c@email.com)"},
            timeout=25  
        )
        ov_resp.raise_for_status()
        
        results = ov_resp.json().get("elements", [])
        hospitals = []
        for el in results:
            tags = el.get("tags", {})
            name = tags.get("name", "").strip()
            if not name:
                continue
            hospitals.append({
                "name": name,
                "address": tags.get("addr:full", tags.get("addr:street", "Address not listed")),
                "phone": tags.get("phone", tags.get("contact:phone", "N/A")),
                "website": tags.get("website", tags.get("contact:website", None)),
                "type": tags.get("amenity", tags.get("healthcare", "Hospital")).title(),
                "lat": el.get("lat"),
                "lon": el.get("lon")
            })
        return hospitals[:5], display_name

    except Exception as e:
        return None, str(e)

def ask_llm(predicted_label: str, question: str, chat_history: list) -> str:
    messages = [
        {
            "role": "system",
            "content": f"""You are RetinaScan AI, a knowledgeable medical assistant specializing in ophthalmology.
The patient's fundus image has been analyzed and the predicted condition is: {predicted_label}.
Give clear, accurate, compassionate guidance. Use simple language. Structure answer with brief points when helpful.
ALWAYS end with: "⚠️ Disclaimer: This is AI-generated information, not a medical diagnosis. Please consult a qualified ophthalmologist." """
        }
    ]
    for role, msg in chat_history[-6:]:
        messages.append({"role": "user" if role == "User" else "assistant", "content": msg})
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct:fastest",
            messages=messages,
            max_tokens=400,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Could not get response from AI: {str(e)}"

    prompt = f"""<s>[INST] You are RetinaScan AI, a knowledgeable medical assistant specializing in ophthalmology.
The patient's fundus image has been analyzed and the predicted condition is: **{predicted_label}**.

Previous conversation:
{history_text}

User's new question: {question}

Instructions:
- Give clear, accurate, compassionate guidance relevant to {predicted_label}
- Use simple language (avoid excessive jargon)
- Structure answer with brief points when helpful
- ALWAYS end with: "⚠️ Disclaimer: This is AI-generated information, not a medical diagnosis. Please consult a qualified ophthalmologist."
[/INST]"""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=400,
            temperature=0.6,
            repetition_penalty=1.1
        )
        return response.strip()
    except Exception as e:
        return f"❌ Could not get response from AI: {str(e)}"

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>👁️</div>
        <div style='font-family:"DM Serif Display",serif; font-size:1.4rem; color:#00d4ff;'>RetinaScan AI</div>
        <div style='color:#94a3b8; font-size:0.8rem;'>Fundus Image Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 About")
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.88rem; line-height:1.6;'>
    RetinaScan AI uses deep learning to screen fundus images for four conditions:
    <br><br>
    🔵 <b>Cataract</b><br>
    🔴 <b>Diabetic Retinopathy</b><br>
    🟡 <b>Glaucoma</b><br>
    🟢 <b>Normal</b>
    <br><br>
    <i>For educational and screening assistance only.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.85rem;'>
    <b>Architecture:</b> DenseNet121 <br>
    <b>Input:</b> 224×224 RGB fundus images<br>
    <b>Classes:</b> 4 (binary per class)<br>
    <b>LLM:</b> Mistral-7B-Instruct
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.78rem; text-align:center;'>
    ⚠️ Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">Retinal Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a fundus image for AI-powered ocular disease screening</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Load model once
model = load_model()

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
col_upload, col_right = st.columns([1.2, 1])

with col_upload:
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Fundus Image")
    uploaded_file = st.file_uploader(
        "Accepted formats: JPG, PNG, JPEG",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="card">
        <div style='font-weight:600; margin-bottom:0.75rem; color:#00d4ff;'>🔬 How It Works</div>
        <div class="step"><span class="step-num">1</span> Upload a fundus (retinal) photograph</div>
        <div class="step"><span class="step-num">2</span> AI model classifies the condition</div>
        <div class="step"><span class="step-num">3</span> Get treatment guidance & specialist info</div>
        <div class="step"><span class="step-num">4</span> Find nearby hospitals in your city</div>
        <div class="step"><span class="step-num">5</span> Ask the AI assistant any questions</div>
    </div>
    """, unsafe_allow_html=True)

# ─── RESULTS ───────────────────────────────────────────────────────────────────
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    predicted_label, confidence, all_probs = classify_image(model, image)
    info = CONDITION_INFO[predicted_label]

    st.markdown("---")
    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.markdown("#### Uploaded Image")
        st.image(image, use_container_width=True)

    with c2:
        st.markdown("#### 🩺 Diagnosis Result")
        st.markdown(f"""
        <div class="card">
            <div style='color:#94a3b8; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;'>Predicted Condition</div>
            <span class="pred-badge {info['badge_class']}">{info['icon']} {predicted_label.replace('_', ' ').title()}</span>
            <div style='margin-top:0.75rem;'>
                <div style='color:#94a3b8; font-size:0.85rem;'>Confidence: <b style='color:#e2e8f0;'>{confidence*100:.1f}%</b></div>
                <div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{confidence*100:.1f}%'></div></div>
            </div>
            <div style='margin-top:1rem; font-size:0.9rem; line-height:1.6; color:#cbd5e1;'>{info['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Probability Breakdown**")
        for cn, p in zip(CLASS_NAMES, all_probs):
            bar_w = float(p) * 100
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;'>
                <div style='width:130px; font-size:0.82rem; color:#94a3b8;'>{cn.replace("_"," ").title()}</div>
                <div style='flex:1; background:#1a2235; border-radius:4px; height:6px;'>
                    <div style='width:{bar_w:.1f}%; background:{"#10b981" if cn=="normal" else "#00d4ff"}; height:6px; border-radius:4px;'></div>
                </div>
                <div style='width:42px; text-align:right; font-size:0.82rem; color:#e2e8f0;'>{bar_w:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ─── TREATMENT STEPS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Recommended Next Steps")
    col_steps, col_life = st.columns(2)

    with col_steps:
        urgency_color = "#ef4444" if info["urgency"].startswith("Urgent") else "#f59e0b" if info["urgency"].startswith("Non") else "#10b981"
        st.markdown(f"""
        <div class="card card-{'danger' if info['severity']=='severe' else 'warning' if info['severity']=='moderate' else 'success'}">
            <div style='margin-bottom:0.75rem;'>
                <div style='font-weight:600; font-size:0.95rem; margin-bottom:0.25rem;'>🏥 Recommended Specialist</div>
                <div style='color:#00d4ff;'>{info['specialist']}</div>
            </div>
            <div style='margin-bottom:0.75rem;'>
                <div style='font-weight:600; font-size:0.95rem; margin-bottom:0.25rem;'>⏱️ Urgency</div>
                <div style='color:{urgency_color};'>{info['urgency']}</div>
            </div>
            <div style='font-weight:600; font-size:0.95rem; margin-bottom:0.5rem;'>📌 Clinical Steps</div>
            {''.join([f"<div class='step'><span class='step-num'>{i+1}</span> {s}</div>" for i, s in enumerate(info['steps'])])}
        </div>
        """, unsafe_allow_html=True)

    with col_life:
        st.markdown(f"""
        <div class="card card-accent">
            <div style='font-weight:600; font-size:0.95rem; margin-bottom:0.75rem; color:#00d4ff;'>🌿 Lifestyle Recommendations</div>
            {''.join([f"<div class='step'><span class='step-num'>✓</span> {tip}</div>" for tip in info['lifestyle']])}
            <div style='margin-top:1.25rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.08); color:#f59e0b; font-size:0.83rem;'>
            ⚠️ <b>Medical Disclaimer:</b> This analysis is for educational screening purposes only and does not constitute a medical diagnosis. Always consult a licensed ophthalmologist.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── HOSPITAL FINDER ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏥 Find Nearby Eye Hospitals")
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;'>
    Enter your city or area to find ophthalmology hospitals and eye clinics near you.
    </div>
    """, unsafe_allow_html=True)

    loc_col1, loc_col2 = st.columns([2, 1])
    with loc_col1:
        user_location = st.text_input(
            "Your location",
            placeholder="e.g. Dehradun, Uttarakhand or Mumbai or 'Bengaluru, Karnataka'",
            label_visibility="collapsed"
        )
    with loc_col2:
        search_hospitals = st.button("🔍 Search Hospitals", use_container_width=True)

    if search_hospitals and user_location:
        with st.spinner("🔍 Searching for nearby eye hospitals..."):
            hospitals, location_display = get_hospitals_nominatim(user_location, predicted_label)

        if hospitals:
            st.markdown(f"""
            <div style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.75rem;'>
            📍 Showing results near: <b style='color:#00d4ff;'>{location_display[:60]}...</b>
            </div>
            """, unsafe_allow_html=True)
            for h in hospitals:
                website_link = f"<a href='{h['website']}' target='_blank' style='color:#00d4ff; text-decoration:none;'>🌐 Website</a>" if h.get("website") else ""
                maps_link = f"<a href='https://www.google.com/maps/search/?api=1&query={h['lat']},{h['lon']}' target='_blank' style='color:#7c3aed; text-decoration:none;'>📍 View on Maps</a>" if h.get("lat") else ""
                st.markdown(f"""
                <div class="doctor-card">
                    <div class="doc-name">🏥 {h['name']}</div>
                    <div class="doc-spec">{h['type']}</div>
                    <div class="doc-meta">📌 {h['address']}</div>
                    <div class="doc-meta">📞 {h['phone']}</div>
                    <div style='margin-top:0.4rem; display:flex; gap:1rem;'>{maps_link} {website_link}</div>
                </div>
                """, unsafe_allow_html=True)
        elif hospitals == []:
            st.info("No hospitals found within 10km. Try a broader city name (e.g. 'Delhi' instead of a specific area).")
        else:
            st.warning(f"Could not geocode location. Try a different format, e.g. 'Dehradun, India'.")

    # ─── CHATBOT ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 AI Medical Assistant")
    st.markdown(f"""
    <div style='color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;'>
    Ask anything about <b style='color:#00d4ff;'>{predicted_label.replace('_', ' ').title()}</b> — symptoms, treatment, lifestyle, medications, follow-ups, and more.
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_condition" not in st.session_state:
        st.session_state.current_condition = ""

    # Reset chat if condition changes
    if st.session_state.current_condition != predicted_label:
        st.session_state.chat_history = []
        st.session_state.current_condition = predicted_label

    # Display history
    if st.session_state.chat_history:
        for role, message in st.session_state.chat_history:
            if role == "User":
                st.markdown(f"""
                <div style='text-align:right;'>
                    <div class='chat-label' style='text-align:right;'>YOU</div>
                    <div class='chat-user'>{message}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                    <div class='chat-label'>🤖 RETINACAN AI</div>
                    <div class='chat-bot'>{message}</div>
                </div>
                """, unsafe_allow_html=True)

    # Suggested questions
    suggestions = {
        "cataract": ["What are early signs of cataract?", "Is cataract surgery safe?", "How long is recovery after surgery?"],
        "diabetic_retinopathy": ["What blood sugar level is safe for my eyes?", "Can diabetic retinopathy be reversed?", "What are anti-VEGF injections?"],
        "glaucoma": ["What causes glaucoma?", "Are glaucoma eye drops permanent?", "Can glaucoma cause blindness?"],
        "normal": ["How often should I get an eye exam?", "What foods are good for eye health?", "How can I prevent eye disease?"]
    }

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**💡 Suggested questions:**")
    sug_cols = st.columns(3)
    for i, sug in enumerate(suggestions[predicted_label]):
        with sug_cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                with st.spinner("Getting AI response..."):
                    response = ask_llm(predicted_label, sug, st.session_state.chat_history)
                st.session_state.chat_history.append(("User", sug))
                st.session_state.chat_history.append(("Bot", response))
                st.rerun()

    # Free-text input
    user_question = st.text_input(
        "Ask your question",
        placeholder=f"Ask about {predicted_label.replace('_', ' ')}...",
        key="chat_input",
        label_visibility="collapsed"
    )
    ask_col1, ask_col2 = st.columns([4, 1])
    with ask_col2:
        send = st.button("Send ➤", use_container_width=True)
    if send and user_question.strip():
        with st.spinner("🤖 Thinking..."):
            response = ask_llm(predicted_label, user_question, st.session_state.chat_history)
        st.session_state.chat_history.append(("User", user_question))
        st.session_state.chat_history.append(("Bot", response))
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

elif uploaded_file and not model:
    st.error("❌ Model could not be loaded. Please check the model path in `MODEL_PATH`.")
else:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem; color:#475569;'>
        <div style='font-size:5rem; margin-bottom:1rem;'>👁️</div>
        <div style='font-size:1.2rem; color:#64748b;'>Upload a fundus image above to begin analysis</div>
        <div style='font-size:0.88rem; margin-top:0.5rem; color:#334155;'>Supports: JPG, PNG, JPEG • Recommended: High-quality fundus photographs</div>
    </div>
    """, unsafe_allow_html=True)
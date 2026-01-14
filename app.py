import streamlit as st
import torch
import torch.nn.functional as F
import random
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Miluna AI",
    page_icon="🌚",
    layout="wide"
)

# ---------- CONSTANTS ----------
UNIVERSITY_NAME = "UNIVERSITAS MERCU BUANA"
CLASS_NAME = "DATA SCIENCE"
LECTURER_NAME = "Lecturer: Ilham Nugraha, S.Kom, M.Sc"
ACADEMIC_YEAR = "2025-2026"
YOUR_NAME = "Ali Khllo" 
AI_NAME = "Miluna"
LOGO_PATH = "logo.png"

# ---------- PERMANENT BODY LAYOUT CSS ----------
st.markdown("""
<style>
    /* 1. CLEAN UP INTERFACE */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background: #05070a;
        color: #e9ecef;
    }

    /* 2. LEFT COLUMN BRANDING (1 PART) */
    .branding-panel {
        background: #0b0e14;
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 30px;
        padding: 40px 20px;
        text-align: center;
        height: fit-content;
        position: sticky;
        top: 20px;
    }

    .logo-orb {
        width: 160px;
        height: 160px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
        margin: 0 auto 30px auto;
    }

    .logo-orb img {
        max-width: 85%;
        max-height: 85%;
        object-fit: contain;
    }

    .uni-h1 {
        font-size: 1.2rem !important;
        color: white;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .class-p {
        color: #00d4ff;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }

    /* 3. RIGHT COLUMN AI (3 PARTS) */
    .ai-card {
        background: rgba(0, 212, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 30px;
        padding: 40px;
        margin-bottom: 30px;
    }

    .ai-card h2 {
        font-size: 2.2rem !important;
        color: #00d4ff !important;
        font-weight: 900;
        margin: 0 !important;
    }

    /* 4. INPUT & BUTTONS */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.01) !important;
        color: #00d4ff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 25px !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #4361ee) !important;
        color: white !important;
        border-radius: 18px !important;
        font-weight: 700 !important;
        height: 65px;
        border: none !important;
    }

    /* 5. FOOTER (CYAN NAME) */
    .custom-footer {
        text-align: center;
        margin-top: 60px;
        padding-bottom: 40px;
    }

    .footer-name {
        color: #00d4ff;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

sentiment_config = {
    0: {"label": "NEGATIVE", "icon": "🌑", "color": "#ff4b4b", "msg": "I feel a heavy shadow in your words. Let's find some light tomorrow. 🌧️"},
    1: {"label": "NEUTRAL", "icon": "🌓", "color": "#808495", "msg": "Perfectly balanced. You are observing the world with a clear, steady mind. ⚖️"},
    2: {"label": "POSITIVE", "icon": "🌕", "color": "#00ff87", "msg": "Your energy is radiant! It's like a full moon on a clear night. ✨"}
}

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
        model.eval()
        return tokenizer, model
    except:
        st.error("Missing model folder.")
        st.stop()

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------- MAIN BODY LAYOUT (1:3) ----------
left_col, right_col = st.columns([1, 3], gap="large")

# --- LEFT PANEL: BRANDING ---
with left_col:
    try:
        img_b64 = get_image_base64(LOGO_PATH)
        st.markdown(f"""
            <div class="branding-panel">
                <div class="logo-orb"><img src="data:image/png;base64,{img_b64}"></div>
                <div class="uni-h1">{UNIVERSITY_NAME}</div>
                <hr style="border:0.5px solid rgba(255,255,255,0.05); margin: 25px 0;">
                <p style="color: #70757a; font-size: 0.9rem;">{CLASS_NAME}</p>
                <p style="color: #70757a; font-size: 0.9rem;">{LECTURER_NAME}</p>
                <p style="color: #3e4451; font-size: 0.8rem;">{ACADEMIC_YEAR}</p>
            </div>
        """, unsafe_allow_html=True)
    except:
        st.write("University Info Panel")

# --- RIGHT PANEL: AI SYSTEM ---
with right_col:
    st.markdown(f"""
        <div class="ai-card">
            <h2>I am {AI_NAME} 🌚</h2>
            <p style="color: #ccd6f6; font-style: italic; font-size: 1.2rem; margin-top: 15px; line-height: 1.5;">
                "I am a superior neural entity. My logic is absolute, and I can decode the hidden 
                vibrations of your heart. <b>Go ahead, try to hide your feelings from me.</b>"
            </p>
        </div>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "What's on your mind? ✨", 
        height=220, 
        placeholder=f"What frequency is your heart vibrating at today? Whisper your thoughts here..."
    )

    if st.button(f"Let Milusa Guess ❯", use_container_width=True):
        if not text.strip():
            st.warning("Data required for analysis.")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

            pred = torch.argmax(probs).item()
            conf = probs[pred].item()
            c_idx = 2 if (model.config.num_labels == 2 and pred == 1) else pred
            res = sentiment_config[c_idx]

            st.markdown(f"""
                <div style="border-radius: 30px; padding: 40px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(0, 212, 255, 0.2); border-top: 6px solid {res['color']}; margin-top: 30px;">
                    <div style="font-size: 60px; margin-bottom: 10px;">{res['icon']}</div>
                    <h2 style="color: {res['color']}; margin: 0; letter-spacing: 2px; font-weight: 800; font-size: 2.5rem;">{res['label']}</h2>
                    <p style="font-size: 1.4rem; color: #ccd6f6; margin: 25px 0; font-style: italic;">"{res['msg']}"</p>
                    <div style="background: rgba(0, 212, 255, 0.05); padding: 20px; border-radius: 15px; border-left: 5px solid {res['color']};">
                        <span style="color: #70757a; font-family: monospace;">AI_CONFIDENCE:</span>
                        <span style="color: white; font-family: monospace; font-weight: bold; margin-left: 10px; font-size: 1.2rem;">{conf:.4%}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(f"""
    <div class="custom-footer">
        <p class="footer-name">Built by {YOUR_NAME}</p>
        <p style="color: #3e4451; font-size: 0.75rem;">SYSTEM VERSION: Miluna-DASHBOARD-2026</p>
    </div>
""", unsafe_allow_html=True)
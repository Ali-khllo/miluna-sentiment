import streamlit as st
import torch
import torch.nn.functional as F
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# 🔧 PERFORMANCE TUNING
torch.set_grad_enabled(False)

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

# ---------- STYLE & 1:3 LAYOUT ----------
st.markdown("""
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {background: #05070a; color: #e9ecef;}

/* Left Branding Column */
.branding-panel {
    background: #0b0e14;
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 30px;
    padding: 40px 20px;
    text-align: center;
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
    margin: 0 auto 30px;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
}

.logo-orb img {max-width: 85%;}

/* Right AI Column */
.ai-card {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 30px;
    padding: 40px;
    margin-bottom: 30px;
}

.ai-card h2 {
    color: #00d4ff !important;
    font-size: 2.2rem !important;
    font-weight: 900;
}

/* Input Fields */
.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    font-size: 1.1rem !important;
}

.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#4361ee) !important;
    color: white !important;
    border-radius: 18px !important;
    height: 65px;
    font-weight: 700 !important;
    border: none !important;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
}

/* Results Display */
.result-box {
    border-radius: 30px;
    padding: 40px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    margin-top: 30px;
}

/* Footer Branding */
.custom-footer {
    text-align: center;
    margin-top: 60px;
    padding-bottom: 40px;
}

.footer-name {
    color: #00d4ff; /* Miluna Cyan */
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 🔥 TWO-LABEL MAPPING
sentiment_config = {
    0: {
        "label": "NEGATIVE", 
        "icon": "🌑", 
        "color": "#ff4b4b", 
        "msg": "I sense a heavy frequency in your consciousness. Logic suggests a period of rest."
    },
    1: {
        "label": "POSITIVE", 
        "icon": "🌕", 
        "color": "#00ff87", 
        "msg": "High-vibration energy detected. Your data stream is illuminating the void."
    }
}

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"AI Engine Offline: Ensure 'model' folder exists. Error: {e}")
        st.stop()

tokenizer, model = load_model()

# ---------- 1:3 RATIO LAYOUT ----------
left_col, right_col = st.columns([1, 3], gap="large")

# --- LEFT: BRANDING ---
with left_col:
    try:
        img = get_image_base64(LOGO_PATH)
        st.markdown(f"""
        <div class="branding-panel">
            <div class="logo-orb"><img src="data:image/png;base64,{img}"></div>
            <b style="color:white; font-size:1.1rem;">{UNIVERSITY_NAME}</b><br><br>
            <span style="color:#00d4ff; font-weight:800;">{CLASS_NAME}</span><br>
            <p style="margin-top:15px; color:#70757a; font-size:0.9rem;">{LECTURER_NAME}</p>
            <p style="color:#3e4451; font-size:0.8rem;">{ACADEMIC_YEAR}</p>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown("### BRANDING PANEL")

# --- RIGHT: AI INTERFACE ---
with right_col:
    st.markdown(f"""
    <div class="ai-card">
        <h2>I am {AI_NAME} 🌚</h2>
        <p style="font-style:italic; color:#ccd6f6; font-size:1.2rem;">
            "Transfer your consciousness into my neural core. I am ready to decode the vibrations hidden in your lines."
        </p>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "What's on your mind? ✨", 
        height=220,
        placeholder="Speak your mind. Nothing escapes my logic..."
    )

    if st.button(f"Let {AI_NAME} Guess ❯", use_container_width=True):
        if not text.strip():
            st.warning("My sensors require data to function. Please input text.")
        else:
            with st.spinner("Decoding vibrations..."):
                # Tokenize and Run Model
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

                # Process Connection to UI
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()
                res = sentiment_config[pred]

                # Display Results
                st.markdown(f"""
                <div class="result-box" style="border-top: 6px solid {res['color']};">
                    <div style="font-size:60px; margin-bottom:10px;">{res['icon']}</div>
                    <h2 style="color:{res['color']}; margin:0; letter-spacing:2px; font-weight:900;">{res['label']}</h2>
                    <p style="font-size:1.3rem; margin:20px 0; font-style:italic; color:#ccd6f6;">"{res['msg']}"</p>
                    <div style="background:rgba(0,212,255,0.08); padding:15px; border-radius:12px; border-left:4px solid {res['color']};">
                        <span style="color:#70757a; font-family:monospace;">LOGIC_CONFIDENCE:</span>
                        <span style="color:white; font-family:monospace; font-weight:bold; margin-left:10px;">{conf:.4%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(f"""
<div class="custom-footer">
    <span class="footer-name">Built by {YOUR_NAME}</span><br>
    <span style="color:#3e4451; font-size:0.8rem;">2026</span>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import torch
import torch.nn.functional as F
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔧 PERFORMANCE
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

# ---------- STYLE ----------
st.markdown("""
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {background: #05070a; color: #e9ecef;}

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
}

.logo-orb img {max-width: 85%;}

.ai-card {
    background: rgba(0,212,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 30px;
    padding: 40px;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.01) !important;
    color: #00d4ff !important;
    border-radius: 20px !important;
}

.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#4361ee) !important;
    color: white !important;
    border-radius: 18px !important;
    height: 65px;
}

.custom-footer {
    text-align: center;
    margin-top: 60px;
    padding-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 🔥 TWO-LABEL CONFIG ONLY
sentiment_config = {
    0: {"label": "NEGATIVE", "icon": "🌑", "color": "#ff4b4b", "msg": "I sense darkness in your words."},
    1: {"label": "POSITIVE", "icon": "🌕", "color": "#00ff87", "msg": "Your energy shines bright!"}
}

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    model.eval()
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# ---------- LAYOUT ----------
left_col, right_col = st.columns([1, 3], gap="large")

# --- LEFT ---
with left_col:
    try:
        img = get_image_base64(LOGO_PATH)
        st.markdown(f"""
        <div class="branding-panel">
            <div class="logo-orb"><img src="data:image/png;base64,{img}"></div>
            <b>{UNIVERSITY_NAME}</b><br><br>
            {CLASS_NAME}<br>
            {LECTURER_NAME}<br>
            {ACADEMIC_YEAR}
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

# --- RIGHT ---
with right_col:
    st.markdown(f"""
    <div class="ai-card">
        <h2>I am {AI_NAME} 🌚</h2>
        <p><i>Try to hide your feelings from me.</i></p>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area("What's on your mind? ✨", height=220)

    if st.button("Let Miluna Guess ❯", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Miluna is thinking..."):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

                pred = torch.argmax(probs).item()
                conf = probs[pred].item()

                res = sentiment_config[pred]

                st.markdown(f"""
                <div style="border-radius:30px;padding:40px;border-top:6px solid {res['color']};">
                    <div style="font-size:60px">{res['icon']}</div>
                    <h2 style="color:{res['color']}">{res['label']}</h2>
                    <p>{res['msg']}</p>
                    <b>CONFIDENCE: {conf:.2%}</b>
                </div>
                """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(f"""
<div class="custom-footer">
    <b>Built by {YOUR_NAME}</b><br>
    SYSTEM VERSION: Miluna-DASHBOARD-2026
</div>
""", unsafe_allow_html=True)

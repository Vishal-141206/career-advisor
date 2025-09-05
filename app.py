import streamlit as st
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Career Advisor AI",
    page_icon="🎓",
    layout="centered"
)

# --- LOAD MODEL & LABEL ENCODER ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('svm_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        st.error("🚨 Critical Error: 'svm_model.pkl' or 'label_encoder.pkl' not found.")
        st.info("Please make sure the model files are in the same folder as the application.")
        return None, None

model, label_encoder = load_artifacts()

# --- QUIZ DATA ---
questions = [
    "🔧 Do you enjoy hands-on activities like fixing gadgets, repairing things, or working with mechanical tools?",
    "🌳 Do you prefer spending time outdoors and being physically active rather than sitting indoors for long hours?",
    "🧩 Do you like solving complex problems, puzzles, or understanding how things work at a deeper level?",
    "🔬 Are you interested in conducting experiments, doing research, or exploring abstract scientific ideas?",
    "🎨 Do you feel fulfilled when you express yourself creatively through art, music, writing, or design?",
    "🕒 Do you enjoy having flexibility in your work schedule rather than following a strict routine?",
    "❤️ Do you feel happy when helping, teaching, or taking care of others?",
    "👂 Are you good at listening to people and helping them resolve conflicts or problems?",
    "👔 Do you enjoy leading teams, persuading others, or taking on business challenges?",
    "🚀 Do you feel ambitious and motivated to take risks in order to achieve bigger goals?",
    "📋 Do you like working in an organized environment with clear rules, predictable tasks, and step-by-step processes?",
    "📊 Do you enjoy managing data, handling budgets, or keeping detailed records and reports?"
]
TOTAL_QUESTIONS = len(questions)

tips = [
    "💪 Hands-on learning builds real skills. Keep experimenting!",
    "🌟 Being active boosts creativity and energy.",
    "🧐 Curiosity fuels discovery. Keep asking questions!",
    "🔍 Research develops deep understanding. Explore!",
    "🎨 Creativity is your superpower. Express yourself!",
    "🕊 Flexibility helps you adapt and grow.",
    "❤️ Helping others creates meaningful impact.",
    "👂 Listening builds trust and strong relationships.",
    "👔 Leadership opens doors. Step up!",
    "🚀 Ambition drives success. Keep aiming high!",
    "📋 Organization helps achieve goals efficiently.",
    "📊 Attention to detail is a powerful skill."
]

dimension_map = [
    "Realistic", "Realistic", "Investigative", "Investigative",
    "Artistic", "Artistic", "Social", "Social",
    "Enterprising", "Enterprising", "Conventional", "Conventional"
]

# --- SESSION STATE INITIALIZATION ---
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CALLBACK FUNCTIONS ---
def next_question():
    st.session_state.answers.append(st.session_state.current_answer)
    st.session_state.current_question += 1

def restart_quiz():
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CSS STYLING (FIXED BUTTON TEXT TO WHITE) ---
theme_css = """
<style>
/* General button style */
.stButton > button {
    background-color: #0066cc !important;
    color: #ffffff !important;   /* ✅ Force white text */
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

/* Hover state */
.stButton > button:hover {
    background-color: #0052a3 !important;
    color: #ffffff !important;   /* ✅ Keep text white */
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,80,180,0.2) !important;
}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- MAIN UI LOGIC ---
if model is None or label_encoder is None:
    st.warning("Application cannot proceed without the necessary model files.")
else:
    # --- HOME PAGE ---
    if not st.session_state.quiz_started:
        st.title("🎓 Welcome to Your Personal Career Advisor")
        st.markdown("Discover your strengths and get personalized guidance...")

        if st.button("🚀 Start Your Journey Now", use_container_width=True):
            st.session_state.quiz_started = True
            st.rerun()

    # --- QUIZ QUESTIONS ---
    elif st.session_state.current_question < TOTAL_QUESTIONS:
        q_num = st.session_state.current_question
        st.markdown(f"### Question {q_num + 1} of {TOTAL_QUESTIONS}")
        progress = int(((q_num) / TOTAL_QUESTIONS) * 100)
        st.progress(progress)

        st.markdown(f"<div><h3>{questions[q_num]}</h3><p>{tips[q_num]}</p></div>", unsafe_allow_html=True)

        st.radio(
            "Your answer:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1:"😡 Strongly Disagree", 2:"🙁 Disagree", 3:"😐 Neutral", 4:"🙂 Agree", 5:"😍 Strongly Agree"}[x],
            key='current_answer',
            horizontal=True
        )
        st.button("Next ➡️", on_click=next_question, use_container_width=True)

    # --- RESULTS PAGE ---
    else:
        with st.spinner("✨ Analyzing your results..."):
            time.sleep(1)

        input_data = np.array([st.session_state.answers])
        prediction_encoded = model.predict(input_data)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]

        st.balloons()
        st.success(f"🎯 Your recommended stream: **{prediction_text}**")

        # --- Restart Quiz Button ---
        if st.button("🔄 Take Quiz Again", use_container_width=True):
            restart_quiz()
            st.rerun()

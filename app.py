import streamlit as st
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

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

# --- SESSION STATE ---
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CALLBACKS ---
def next_question():
    st.session_state.answers.append(st.session_state.current_answer)
    st.session_state.current_question += 1

def restart_quiz():
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CSS for fade-in cards ---
fade_css = """
<style>
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.fade-card {
    animation: fadeIn 0.8s ease-in-out;
    border-radius: 20px;
    padding: 25px;
    background: linear-gradient(135deg, #d0f0fd, #a0d8ef);
    box-shadow: 0 8px 16px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}
.fade-card h3 {
    color: #1f77b4;
    margin-bottom: 10px;
}
.fade-card p {
    color: #0b3d91;
    font-size: 16px;
    font-weight: 600;
}
</style>
"""
st.markdown(fade_css, unsafe_allow_html=True)

# --- MAIN UI ---
if model is None or label_encoder is None:
    st.error("🚨 Model or label encoder missing. Ensure 'svm_model.pkl' and 'label_encoder.pkl' are present.")
else:
    if not st.session_state.quiz_started:
        st.title("🎓 Welcome to Your Personal Career Advisor")
        st.markdown(
            "In the competitive landscape of modern education, making the right choice after Class 10th or 12th is crucial. "
            "This AI-powered tool helps you discover your strengths and suggests the best educational stream and career path."
        )
        st.info("✨ \"The best way to predict the future is to create it.\" — Abraham Lincoln")
        st.divider()
        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_started = True
            st.rerun()

    elif st.session_state.current_question < TOTAL_QUESTIONS:
        q_num = st.session_state.current_question
        st.markdown(f"### Question {q_num+1} of {TOTAL_QUESTIONS}")

        # Progress bar
        progress = int((q_num / TOTAL_QUESTIONS) * 100)
        st.progress(progress)

        st.markdown(
            f"""
            <div class="fade-card">
                <h3>{questions[q_num]}</h3>
                <p>{tips[q_num]}</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.radio(
            "Your answer:",
            options=[1,2,3,4,5],
            format_func=lambda x: {1:"😡 Strongly Disagree", 2:"🙁 Disagree", 3:"😐 Neutral", 4:"🙂 Agree", 5:"😍 Strongly Agree"}[x],
            key='current_answer',
            horizontal=True
        )

        st.button("Next ➡️", on_click=next_question, use_container_width=True)

    else:
        with st.spinner("✨ Analyzing your results..."):
            time.sleep(1)

        input_data = np.array([st.session_state.answers])
        prediction_encoded = model.predict(input_data)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]

        st.balloons()
        st.success(f"🎯 Recommended Stream: **{prediction_text}**")

        degree_map = {
            "Science": ["B.Tech in CS", "B.Sc Physics", "MBBS", "BCA"],
            "Commerce": ["B.Com Hons", "CA", "BBA Finance", "BA Economics"],
            "Arts": ["BA Psychology", "BFA", "BA Journalism", "BA English Lit"],
            "Vocational": ["Diploma Web Designing", "ITI Electrical", "B.Voc Hospitality", "Skill Plumbing"]
        }
        career_map = {
            "Science": ["Software Engineer", "Research Scientist", "Doctor", "Data Scientist"],
            "Commerce": ["Accountant", "Investment Banker", "Entrepreneur", "Financial Analyst"],
            "Arts": ["Psychologist", "Graphic Designer", "Journalist", "Content Writer"],
            "Vocational": ["Full-Stack Dev", "Electrician", "Hotel Manager", "Mechanic"]
        }

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("🎓 Degree Options")
            for d in degree_map.get(prediction_text, ["N/A"]):
                st.markdown(f"✔️ {d}")
        with col2:
            st.subheader("💼 Career Paths")
            for c in career_map.get(prediction_text, ["N/A"]):
                st.markdown(f"✔️ {c}")

        st.divider()

        # --- Radar Chart without Icons ---
        dimension_scores = {}
        for i, dim in enumerate(dimension_map):
            dimension_scores[dim] = dimension_scores.get(dim, 0) + st.session_state.answers[i]

        labels = list(dimension_scores.keys())
        scores = list(dimension_scores.values())
        scores += scores[:1]  # close polygon

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        # Colors for edges
        colors = ['#FF6F61','#6B5B95','#88B04B','#F7CAC9','#92A8D1','#955251'][:len(labels)]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

        # Draw polygon
        ax.plot(angles, scores, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, scores, color="#1f77b4", alpha=0.25)

        # Optional: draw edges in different colors
        for i in range(len(labels)):
            ax.plot([angles[i], angles[i+1]], [scores[i], scores[i+1]], color=colors[i], linewidth=2)

        # Set labels (no icons)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)

        ax.set_ylim(0, max(scores)+1)
        ax.set_title("Your Interest Profile", fontsize=16, pad=20)
        ax.grid(True)

        st.pyplot(fig)

        # Restart quiz
        if st.button("🔄 Take Quiz Again", use_container_width=True):
            restart_quiz()
            st.rerun()

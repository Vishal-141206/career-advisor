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
        st.success(f"🎯 Congratulations! Your recommended stream is: **{prediction_text}**")

        # Motivational message
        stream_messages = {
            "Science": "🔬 Explore, experiment, and innovate! Your curiosity will lead you far.",
            "Commerce": "💰 Numbers and strategy are your allies. Time to build your empire!",
            "Arts": "🎨 Creativity and expression are your strengths. Share your vision with the world!",
            "Vocational": "🛠 Hands-on skills open doors. Master your craft and shine!"
        }
        st.info(stream_messages.get(prediction_text, "✨ Explore your interests and shape your future!"))

        # Degree and career options with icons
        degree_map = {
            "Science": [("B.Tech in CS", "💻"), ("B.Sc Physics", "⚛️"), ("MBBS", "🩺"), ("BCA", "🖥")],
            "Commerce": [("B.Com Hons", "📚"), ("CA", "🧾"), ("BBA Finance", "💹"), ("BA Economics", "💵")],
            "Arts": [("BA Psychology", "🧠"), ("BFA", "🎨"), ("BA Journalism", "📰"), ("BA English Lit", "📖")],
            "Vocational": [("Diploma Web Designing", "💻"), ("ITI Electrical", "⚡"), ("B.Voc Hospitality", "🏨"), ("Skill Plumbing", "🔧")]
        }
        career_map = {
            "Science": [("Software Engineer", "💻"), ("Research Scientist", "🔬"), ("Doctor", "🩺"), ("Data Scientist", "📊")],
            "Commerce": [("Accountant", "📒"), ("Investment Banker", "💰"), ("Entrepreneur", "🚀"), ("Financial Analyst", "📈")],
            "Arts": [("Psychologist", "🧠"), ("Graphic Designer", "🎨"), ("Journalist", "📰"), ("Content Writer", "✍️")],
            "Vocational": [("Full-Stack Dev", "💻"), ("Electrician", "⚡"), ("Hotel Manager", "🏨"), ("Mechanic", "🔧")]
        }

        st.subheader("🎓 Degree Options")
        for d, icon in degree_map.get(prediction_text, [("N/A","❓")]):
            with st.expander(f"{icon} {d}"):
                st.write(f"Learn more about **{d}**. This program aligns with your interests in {prediction_text}.")

        st.subheader("💼 Career Paths")
        for c, icon in career_map.get(prediction_text, [("N/A","❓")]):
            with st.expander(f"{icon} {c}"):
                st.write(f"Explore the path of a **{c}**. This career fits well with your strengths in {prediction_text}.")

        st.divider()

        # --- Interactive Radar Chart using Plotly ---
        dimension_scores = {}
        for i, dim in enumerate(dimension_map):
            dimension_scores[dim] = dimension_scores.get(dim, 0) + st.session_state.answers[i]

        labels = list(dimension_scores.keys())
        scores = list(dimension_scores.values())
        scores += scores[:1]
        labels += labels[:1]

        hover_text = [f"{dim}: {st.session_state.answers[i]} — {tips[i]}" for i, dim in enumerate(dimension_map)]
        hover_text += hover_text[:1]

        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=scores,
                    theta=labels,
                    fill='toself',
                    name='Interest Score',
                    hoverinfo='text',
                    hovertext=hover_text,
                    line=dict(color='#1f77b4', width=3)
                )
            ]
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(scores)+1])
            ),
            showlegend=False,
            title="📊 Your Interest Profile"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Restart quiz
        if st.button("🔄 Take Quiz Again", use_container_width=True):
            restart_quiz()
            st.rerun()

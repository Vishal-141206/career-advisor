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
# Use st.cache_resource to load the model and encoder only once.
@st.cache_resource
def load_artifacts():
    try:
        # These files should be in the same directory as your app.py
        model = joblib.load('svm_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        st.error("🚨 Critical Error: 'svm_model.pkl' or 'label_encoder.pkl' not found.")
        st.info("Please make sure the model files are in the same folder as the application.")
        return None, None

model, label_encoder = load_artifacts()

# --- QUIZ DATA ---
# A list of questions for the career quiz.
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

# Motivational tips corresponding to each question.
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

# Mapping each question to one of the six Holland Code dimensions.
dimension_map = [
    "Realistic", "Realistic", "Investigative", "Investigative",
    "Artistic", "Artistic", "Social", "Social",
    "Enterprising", "Enterprising", "Conventional", "Conventional"
]

# --- SESSION STATE INITIALIZATION ---
# This ensures that the quiz state is maintained across reruns.
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CALLBACK FUNCTIONS ---
# These functions modify the session state.
def next_question():
    """Appends the current answer and increments the question number."""
    st.session_state.answers.append(st.session_state.current_answer)
    st.session_state.current_question += 1

def restart_quiz():
    """Resets the session state to start the quiz over."""
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.quiz_started = False

# --- CSS STYLING ---
# Defines the visual theme of the application.
theme_css = """
<style>
body, .stApp {
    background-color: #e6f0ff !important;
}

@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }

.fade-card {
    animation: fadeIn 0.8s ease-in-out;
    border-radius: 20px;
    padding: 25px;
    background: linear-gradient(135deg, #ffffff, #cce0ff);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.fade-card h3 { color: #003366; } /* Card question text */
.fade-card p { color: #004080; font-weight: 600; } /* Card tip text */

.stButton>button {
    background-color: #004080;
    color: white !important; /* FIX: Ensures button text is always white */
    font-weight: 600;
    border-radius: 12px;
    padding: 8px 24px;
    border: none;
}
.stButton>button:hover {
    background-color: #0059b3;
    color: white !important; /* Also ensure hover text is white */
}

div[data-testid="stProgressBar"]>div>div>div>div {
    background-color: #004080 !important;
}

/* Alerts */
.stAlert > div {
    color: #003366 !important;
    background-color: #cce0ff !important;
    border-left: 6px solid #004080 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
}

/* Markdown text styling for better hierarchy */
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #003366 !important; /* Darker blue for headers */
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: #004080 !important; /* Slightly lighter blue for body text */
}

/* Expander header text */
div[role="button"] > div > div > div {
    color: #003366 !important;
}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- MAIN UI LOGIC ---
if model is None or label_encoder is None:
    # If model loading fails, the app stops here.
    st.warning("Application cannot proceed without the necessary model files.")
else:
    # --- HOME PAGE ---
    if not st.session_state.quiz_started:
        st.title("🎓 Welcome to Your Personal Career Advisor")
        st.markdown(
            "Discover your strengths and get personalized guidance for your educational stream and career path."
        )
        st.info("✨ \"The best way to predict the future is to create it.\" — Abraham Lincoln")
        st.divider()
        if st.button("🚀 Start Quiz", use_container_width=True):
            st.session_state.quiz_started = True
            st.rerun()

    # --- QUIZ QUESTIONS ---
    elif st.session_state.current_question < TOTAL_QUESTIONS:
        q_num = st.session_state.current_question
        st.markdown(f"### Question {q_num + 1} of {TOTAL_QUESTIONS}")
        progress = int(((q_num) / TOTAL_QUESTIONS) * 100)
        st.progress(progress)

        # Display question and tip in a styled card.
        st.markdown(
            f"<div class='fade-card'><h3>{questions[q_num]}</h3><p>{tips[q_num]}</p></div>",
            unsafe_allow_html=True
        )

        # Radio buttons for the answer.
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

        # Predict the stream using the loaded SVM model.
        input_data = np.array([st.session_state.answers])
        prediction_encoded = model.predict(input_data)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]

        st.balloons()
        st.success(f"🎯 Your recommended stream: **{prediction_text}**")

        stream_messages = {
            "Science": "🔬 Explore, experiment, and innovate! Your curiosity will lead you far.",
            "Commerce": "💰 Numbers and strategy are your allies. Time to build your empire!",
            "Arts": "🎨 Creativity and expression are your strengths. Share your vision with the world!",
            "Vocational": "🛠 Hands-on skills open doors. Master your craft and shine!"
        }
        st.info(stream_messages.get(prediction_text, "✨ Explore your interests and shape your future!"))

        # --- Degree & Career Options ---
        degree_map = {
            "Science": [("B.Tech in CS", "💻"), ("MBBS", "🩺"), ("B.Sc Physics", "⚛️")],
            "Commerce": [("B.Com Hons", "📚"), ("Chartered Accountancy (CA)", "🧾"), ("BBA", "📈")],
            "Arts": [("BA in Psychology", "🧠"), ("Bachelor of Fine Arts (BFA)", "🎨"), ("Journalism & Mass Comm.", "📰")],
            "Vocational": [("ITI in Electrical", "⚡"), ("B.Voc in Hospitality", "🏨"), ("Diploma in Web Designing", "🌐")]
        }

        career_map = {
            "Science": [("Software Engineer", "💻"), ("Doctor", "🩺"), ("Research Scientist", "🔬")],
            "Commerce": [("Accountant", "📒"), ("Entrepreneur", "🚀"), ("Financial Analyst", "💹")],
            "Arts": [("Psychologist", "🧠"), ("Graphic Designer", "🎨"), ("Journalist", "🎤")],
            "Vocational": [("Electrician", "⚡"), ("Mechanic", "🔧"), ("Chef", "👨‍🍳")]
        }

        st.subheader("🎓 Potential Degree Options")
        for d, icon in degree_map.get(prediction_text, [("N/A", "❓")]):
            with st.expander(f"{icon} {d}"):
                st.write(f"Learn more about **{d}**. This path aligns well with your interests in the {prediction_text} stream.")

        st.subheader("💼 Possible Career Paths")
        for c, icon in career_map.get(prediction_text, [("N/A", "❓")]):
            with st.expander(f"{icon} {c}"):
                st.write(f"Explore the path of a **{c}**. This career fits the strengths typically found in the {prediction_text} stream.")

        st.divider()

        # --- Radar Chart Visualization ---
        dimension_scores = {dim: 0 for dim in set(dimension_map)}
        for i, dim in enumerate(dimension_map):
            dimension_scores[dim] += st.session_state.answers[i]

        labels = list(dimension_scores.keys())
        scores = list(dimension_scores.values())

        # Correctly generate hover text for the aggregated dimension scores.
        hover_text = [f"Total Score: {score}" for score in scores]

        # Ensure the chart loop closes by appending the first item to the end.
        scores_loop = scores + [scores[0]]
        labels_loop = labels + [labels[0]]
        hover_text_loop = hover_text + [hover_text[0]]

        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=scores_loop,
                    theta=labels_loop,
                    fill='toself',
                    fillcolor='rgba(0,64,128,0.3)',
                    line=dict(color='#004080', width=3, shape='linear'),
                    marker=dict(size=10, color='#004080'),
                    hoverinfo='theta+text',
                    hovertext=hover_text_loop
                )
            ]
        )

        fig.update_layout(
            polar=dict(
                bgcolor='#e6f0ff',
                radialaxis=dict(
                    visible=True,
                    range=[0, 10], # Max score for 2 questions (5+5) is 10.
                    gridcolor='rgba(0,64,128,0.2)',
                    linecolor='rgba(0,64,128,0.6)',
                    tickfont=dict(color='#004080', size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(color='#004080', size=13, weight='bold')
                )
            ),
            showlegend=False,
            title=dict(text="📊 Your Interest Profile", font=dict(size=24, color="#003366")),
            paper_bgcolor='#e6f0ff',
            plot_bgcolor='#e6f0ff',
            margin=dict(l=60, r=60, t=80, b=60)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Restart Quiz Button ---
        if st.button("🔄 Take Quiz Again", use_container_width=True):
            restart_quiz()
            st.rerun()


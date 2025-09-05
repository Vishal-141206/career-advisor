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

# --- CSS STYLING ---
theme_css = """
<style>
body, .stApp {
    background-color: #f0f8ff !important;
    color: #003366 !important;
}

/* Buttons */
.stButton>button {
    background-color: #0066cc !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover,
.stButton>button:focus,
.stButton>button:active {
    background-color: #0052a3 !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,80,180,0.2) !important;
}

/* Special primary button */
.stButton.primary>button {
    background-color: #ff6600 !important;
    color: #ffffff !important;
    font-weight: 700;
    border-radius: 14px;
    padding: 12px 26px;
    border: none;
}

.stButton.primary>button:hover {
    background-color: #e65c00 !important;
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(230,92,0,0.3);
}

/* Rest of your CSS (same as before)... */
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
.fade-card {
    animation: fadeIn 0.8s ease-in-out;
    border-radius: 20px;
    padding: 25px;
    background: linear-gradient(135deg, #ffffff, #e6f2ff);
    box-shadow: 0 8px 20px rgba(0,80,180,0.15);
    margin-bottom: 20px;
    border: 1px solid #cce0ff;
}
.fade-card h3 { color: #003366 !important; margin-bottom: 15px; }
.fade-card p { color: #00509e !important; font-weight: 600; margin-bottom: 0; font-size: 0.95rem; }

/* Alerts */
.stAlert > div {
    color: #003366 !important;
    background-color: #e6f2ff !important;
    border-left: 6px solid #0066cc !important;
    border-radius: 10px !important;
    padding: 15px 20px !important;
}

/* Headers */
h1, h2, h3 { color: #003366 !important; }

/* Radio buttons */
div[role="radiogroup"] label { color: #003366 !important; }
div[role="radiogroup"] label:hover { color: #0066cc !important; }

/* Expander styling */
.streamlit-expanderHeader {
    color: #003366 !important;
    font-weight: 600;
    background-color: #e6f2ff;
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 5px;
    border: 1px solid #cce0ff;
}
.streamlit-expanderContent {
    background-color: #f7fbff;
    border-radius: 0 0 8px 8px;
    padding: 15px;
    border: 1px solid #cce0ff;
    border-top: none;
}

/* Progress bar container */
div[data-testid="stProgressBar"] { margin-bottom: 25px; }
div[data-testid="stProgressBar"]>div>div>div>div { background-color: #0066cc !important; }

/* Success & Info */
.stSuccess { background-color: #e6f7ff !important; color: #003366 !important; border-left: 6px solid #0066cc !important; }
.stInfo { background-color: #e6f2ff !important; color: #003366 !important; border-left: 6px solid #0066cc !important; }

/* Markdown text */
div[data-testid="stMarkdownContainer"] p { color: #003366 !important; }

/* Divider */
hr { border-top: 2px solid #cce0ff !important; margin: 30px 0 !important; }

/* Motivational section */
.motivation-card {
    background: linear-gradient(135deg, #e6f2ff, #cce6ff);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    border-left: 5px solid #0066cc;
}
.quote-text { font-style: italic; color: #004080; font-size: 1.1rem; margin-bottom: 10px; }
.quote-author { font-weight: 600; color: #0066cc; text-align: right; }

/* Feature cards */
.feature-grid { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }
.feature-card {
    flex: 1 1 calc(33.333% - 15px);
    min-width: 200px;
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align: center;
    border: 1px solid #cce0ff;
}
.feature-icon { font-size: 2rem; margin-bottom: 10px; }
.feature-title { font-weight: 600; color: #0066cc; margin-bottom: 8px; }
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
        st.markdown(
            "Discover your strengths and get personalized guidance for your educational stream and career path. Move beyond confusion and doubt with a data-driven plan tailored just for you. Make confident decisions today that will shape a successful and fulfilling tomorrow."
        )
        
        # Motivational Section
        st.markdown("""
        <div class="motivation-card">
            <div class="quote-text">"The future depends on what you do today."</div>
            <div class="quote-author">— Mahatma Gandhi</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Grid
        st.subheader("🌟 What You'll Discover")
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Your Strengths</div>
                <div>Identify your natural talents and abilities</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Career Matches</div>
                <div>Find careers that align with your personality</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <div class="feature-title">Education Paths</div>
                <div>Discover the right educational streams for you</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How It Works
        with st.expander("ℹ️ How It Works"):
            st.markdown("""
            1. **Answer 12 simple questions** about your preferences and interests  
            2. **Our AI analyzes** your responses using the Holland Code model  
            3. **Get personalized recommendations** for educational streams and careers  
            4. **Explore your interest profile** with our interactive radar chart
            """)
        
        # Additional Motivation
        st.markdown("""
        <div class="motivation-card">
            <div class="quote-text">"Success is not final, failure is not fatal: It is the courage to continue that counts."</div>
            <div class="quote-author">— Winston Churchill</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        if st.button("🚀 Start Your Journey Now", use_container_width=True):
            st.session_state.quiz_started = True
            st.rerun()
    
    # --- QUIZ QUESTIONS ---
    elif st.session_state.current_question < TOTAL_QUESTIONS:
        q_num = st.session_state.current_question
        st.markdown(f"### Question {q_num + 1} of {TOTAL_QUESTIONS}")
        progress = int(((q_num) / TOTAL_QUESTIONS) * 100)
        st.progress(progress)
        
        st.markdown(
            f"<div class='fade-card'><h3>{questions[q_num]}</h3><p>{tips[q_num]}</p></div>",
            unsafe_allow_html=True
        )
        
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
        
        stream_messages = {
            "Science": "🔬 Explore, experiment, and innovate! Your curiosity will lead you far.",
            "Commerce": "💰 Numbers and strategy are your allies. Time to build your empire!",
            "Arts": "🎨 Creativity and expression are your strengths. Share your vision with the world!",
            "Vocational": "🛠 Hands-on skills open doors. Master your craft and shine!"
        }
        st.info(stream_messages.get(prediction_text, "✨ Explore your interests and shape your future!"))
        
        # Final motivational quote
        st.markdown("""
        <div class="motivation-card">
            <div class="quote-text">"Your time is limited, so don't waste it living someone else's life."</div>
            <div class="quote-author">— Steve Jobs</div>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        hover_text = [f"Total Score: {score}" for score in scores]
        scores_loop = scores + [scores[0]]
        labels_loop = labels + [labels[0]]
        hover_text_loop = hover_text + [hover_text[0]]
        
        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=scores_loop,
                    theta=labels_loop,
                    fill='toself',
                    fillcolor='rgba(0,102,204,0.3)',
                    line=dict(color='#0066cc', width=3, shape='linear'),
                    marker=dict(size=10, color='#0066cc'),
                    hoverinfo='theta+text',
                    hovertext=hover_text_loop
                )
            ]
        )
        
        fig.update_layout(
            polar=dict(
                bgcolor='#f0f8ff',
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    gridcolor='rgba(0,102,204,0.2)',
                    linecolor='rgba(0,102,204,0.6)',
                    tickfont=dict(color='#003366', size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(color='#003366', size=13, weight='bold')
                )
            ),
            showlegend=False,
            title=dict(text="📊 Your Interest Profile", font=dict(size=24, color="#003366")),
            paper_bgcolor='#f0f8ff',
            plot_bgcolor='#f0f8ff',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Restart Quiz Button ---
        if st.button("🔄 Take Quiz Again", use_container_width=True, type="primary"):
            restart_quiz()
            st.rerun()

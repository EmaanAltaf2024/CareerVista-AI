import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CareerVista AI",
    page_icon="🎓",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "careervista_pk_starter_dataset.csv"


# =========================================================
# SIDEBAR AND THEME
# =========================================================
if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = "Dark"

with st.sidebar:
    st.title("🎓 CareerVista AI")
    st.caption("AI-powered career exploration")

    theme = st.selectbox(
        "Interface Theme",
        ["Dark", "Light"],
        key="theme_choice",
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Resume Dashboard",
            "Explore Jobs",
            "Job Records",
            "Model & Project Info",
        ],
    )

dark = theme == "Dark"

if dark:
    bg = "#0b1220"
    panel = "#111c2e"
    fg = "#f1f5f9"
    muted = "#a8b3c7"
    border = "#29364b"
    input_bg = "#111c2e"
    input_border = "#455a78"
    card_hover = "#1b2b43"
else:
    bg = "#f5f7fb"
    panel = "#ffffff"
    fg = "#172033"
    muted = "#536176"
    border = "#dce3ed"
    input_bg = "#ffffff"
    input_border = "#c7d2e2"
    card_hover = "#f0f5ff"



# =========================================================
# GLOBAL STYLING
# =========================================================
dark_uploader_css = ""

if dark:
    dark_uploader_css = """
    div[data-testid="stFileUploader"] section {
        background: #111c2e !important;
        border: 1px dashed #617897 !important;
        color: #f1f5f9 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: #263b56 !important;
        color: #ffffff !important;
        border: 1px solid #8ca8cc !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #385477 !important;
        color: #ffffff !important;
        border-color: #c1d9ff !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {
        color: #d7e2f2 !important;
    }
    """

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {fg};
    }}

    [data-testid="stSidebar"] {{
        background: {panel};
    }}

    /* Only make the CareerVista AI sidebar title white */
    [data-testid="stSidebar"] h1 {{
        color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    h1, h2, h3, h4, p, label {{
        color: {fg};
    }}

    div[data-testid="stMetric"] {{
        background: {panel};
        border: 1px solid {border};
        padding: 14px;
        border-radius: 14px;
    }}

    div[data-testid="stFileUploader"] {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 12px;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 10px;
    }}

    small, .stCaption {{
        color: {muted} !important;
    }}

    .stButton > button {{
        border-radius: 10px;
    }}

    /* Readable search inputs and resume text area in both themes */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {{
        background-color: {input_bg} !important;
        color: {fg} !important;
        -webkit-text-fill-color: {fg} !important;
        caret-color: {fg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 10px !important;
        opacity: 1 !important;
        font-size: 15px !important;
    }}

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {{
        color: {muted} !important;
        -webkit-text-fill-color: {muted} !important;
        opacity: 1 !important;
    }}

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: #8b83f5 !important;
        box-shadow: 0 0 0 2px rgba(139, 131, 245, 0.22) !important;
        outline: none !important;
    }}

    /* Preserve readable text when browser autofill is active */
    div[data-testid="stTextInput"] input:-webkit-autofill,
    div[data-testid="stTextInput"] input:-webkit-autofill:hover,
    div[data-testid="stTextInput"] input:-webkit-autofill:focus {{
        -webkit-text-fill-color: {fg} !important;
        box-shadow: 0 0 0 1000px {input_bg} inset !important;
        transition: background-color 9999s ease-in-out 0s;
    }}

    /* Interactive cards for the project information page */
    .project-card {{
        background: {panel};
        color: {fg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 21px;
        margin-bottom: 16px;
        min-height: 155px;
        transition:
            transform 0.22s ease,
            box-shadow 0.22s ease,
            border-color 0.22s ease,
            background-color 0.22s ease;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
    }}

    .project-card:hover {{
        transform: translateY(-5px);
        background: {card_hover};
        border-color: #8b83f5;
        box-shadow: 0 12px 28px rgba(100, 110, 180, 0.18);
    }}

    .project-card .card-icon {{
        font-size: 27px;
        margin-bottom: 10px;
    }}

    .project-card h3 {{
        color: {fg} !important;
        font-size: 17px;
        margin: 0 0 9px 0;
    }}

    .project-card p {{
        color: {muted} !important;
        font-size: 14px;
        line-height: 1.65;
        margin: 0;
    }}

    .project-intro {{
        background: {panel};
        border: 1px solid {border};
        border-left: 4px solid #8b83f5;
        border-radius: 14px;
        padding: 19px 22px;
        margin: 8px 0 22px 0;
        transition: box-shadow 0.22s ease, transform 0.22s ease;
    }}

    .project-intro:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(100, 110, 180, 0.14);
    }}

    .project-intro p {{
        color: {muted} !important;
        line-height: 1.75;
        margin: 0;
    }}

    .project-section-label {{
        color: {muted};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin: 23px 0 12px 0;
    }}

    /* Readable download buttons in both themes */
    div[data-testid="stDownloadButton"] button {{
        background-color: #5b4bdb !important;
        color: #ffffff !important;
        border: 1px solid #8b83f5 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }}

    div[data-testid="stDownloadButton"] button p {{
        color: #ffffff !important;
    }}

    div[data-testid="stDownloadButton"] button:hover {{
        background-color: #4636bd !important;
        color: #ffffff !important;
        border-color: #b6afff !important;
    }}

    {dark_uploader_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# ANIMATED HERO
# =========================================================
def show_hero(current_theme):
    if current_theme == "Dark":
        hero_background = "#070d1b"
        hero_text = "#f8fbff"
        hero_subtext = "#bdcbe0"
        hero_border = "rgba(132, 160, 210, 0.24)"
    else:
        hero_background = "#f5f7fb"
        hero_text = "#172033"
        hero_subtext = "#536176"
        hero_border = "rgba(100, 120, 150, 0.25)"

    html = r"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        * { box-sizing: border-box; }

        html, body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Inter, Arial, Helvetica, sans-serif;
        }

        .hero {
            position: relative;
            isolation: isolate;
            overflow: hidden;
            min-height: 290px;
            padding: 38px 34px;
            border-radius: 23px;
            background: HERO_BACKGROUND;
            color: HERO_TEXT;
            border: 1px solid HERO_BORDER;
        }

        .ambient {
            position: absolute;
            inset: -45%;
            z-index: -3;
            pointer-events: none;
            background:
                radial-gradient(ellipse at 18% 28%,
                    rgba(88, 83, 255, 0.36), transparent 38%),
                radial-gradient(ellipse at 78% 68%,
                    rgba(0, 224, 205, 0.22), transparent 37%),
                radial-gradient(ellipse at 60% 12%,
                    rgba(178, 83, 255, 0.18), transparent 34%);
            animation: ambientDrift 12s ease-in-out infinite alternate;
            will-change: transform;
        }

        .ambient-two {
            position: absolute;
            inset: -25%;
            z-index: -2;
            pointer-events: none;
            background:
                conic-gradient(from 0deg at 55% 50%,
                    transparent 0deg,
                    rgba(91, 109, 255, 0.12) 90deg,
                    transparent 150deg,
                    rgba(0, 221, 204, 0.10) 240deg,
                    transparent 360deg);
            filter: blur(20px);
            animation: slowRotate 28s linear infinite;
            will-change: transform;
        }

        .grid {
            position: absolute;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            opacity: 0.20;
            background-image:
                linear-gradient(rgba(145, 166, 210, 0.18) 1px,
                    transparent 1px),
                linear-gradient(90deg, rgba(145, 166, 210, 0.18) 1px,
                    transparent 1px);
            background-size: 34px 34px;
            mask-image: linear-gradient(110deg, black, transparent 90%);
        }

        .glow {
            position: absolute;
            width: 190px;
            height: 190px;
            right: 9%;
            top: 28px;
            border-radius: 50%;
            pointer-events: none;
            background: rgba(91, 94, 255, 0.13);
            box-shadow: 0 0 50px rgba(91, 94, 255, 0.18),
                        0 0 110px rgba(0, 221, 204, 0.10);
            filter: blur(12px);
            animation: glowPulse 5s ease-in-out infinite alternate;
        }

        .orb {
            position: absolute;
            border-radius: 50%;
            pointer-events: none;
            filter: blur(1px);
        }

        .orb-one {
            width: 76px;
            height: 76px;
            right: 16%;
            top: 36px;
            border: 1px solid rgba(140, 160, 255, 0.40);
            background: rgba(105, 105, 255, 0.08);
            box-shadow: 0 0 35px rgba(102, 102, 255, 0.25);
            animation: floatOne 7s ease-in-out infinite;
        }

        .orb-two {
            width: 27px;
            height: 27px;
            right: 31%;
            bottom: 34px;
            background: rgba(71, 233, 216, 0.22);
            box-shadow: 0 0 30px rgba(71, 233, 216, 0.42);
            animation: floatTwo 5s ease-in-out infinite;
        }

        .orb-three {
            width: 12px;
            height: 12px;
            right: 7%;
            top: 56%;
            background: #a59bff;
            box-shadow: 0 0 20px #8d83ff;
            animation: floatOne 6s ease-in-out infinite reverse;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            pointer-events: none;
            background: #91fff0;
            box-shadow: 0 0 12px #52dfd0;
            animation: sparkle 3s ease-in-out infinite alternate;
        }

        .p1 { top: 21%; left: 72%; }
        .p2 { top: 72%; left: 83%; animation-delay: 0.6s; }
        .p3 { top: 34%; left: 94%; animation-delay: 1.2s; }
        .p4 { top: 85%; left: 63%; animation-delay: 0.9s; }
        .p5 { top: 16%; left: 56%; animation-delay: 1.7s; }

        .content {
            position: relative;
            z-index: 5;
            max-width: 730px;
            animation: fadeInUp 1s ease-out both;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            margin-bottom: 17px;
            border: 1px solid rgba(130, 160, 220, 0.34);
            border-radius: 999px;
            background: rgba(110, 135, 190, 0.10);
            color: #8ff3e7;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1.8px;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            display: inline-block;
            border-radius: 50%;
            background: #59ebd7;
            box-shadow: 0 0 12px #59ebd7;
            animation: statusPulse 1.8s ease-in-out infinite;
        }

        h1 {
            color: HERO_TEXT;
            margin: 0 0 14px;
            font-size: clamp(30px, 4.2vw, 47px);
            line-height: 1.12;
            letter-spacing: -1.5px;
            font-weight: 800;
            animation: floatTitle 4.6s ease-in-out infinite;
            will-change: transform;
        }

        .gradient-text {
            display: inline-block;
            color: #8e94ff;
            background: linear-gradient(100deg,
                #a29aff 0%, #65eee0 35%,
                #b9a2ff 65%, #a29aff 100%);
            background-size: 240% auto;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 18px rgba(116, 126, 255, 0.22));
            animation: gradientFlow 5s linear infinite;
        }

        .description {
            max-width: 620px;
            margin: 0;
            color: HERO_SUBTEXT;
            font-size: 14px;
            line-height: 1.8;
        }

        .bottom-line {
            position: absolute;
            height: 2px;
            width: 38%;
            bottom: 0;
            left: 0;
            pointer-events: none;
            background: linear-gradient(90deg,
                transparent, #8585ff, #54e8d7, transparent);
            opacity: 0.85;
            filter: blur(0.4px);
            animation: lineMove 7s ease-in-out infinite alternate;
        }

        @keyframes ambientDrift {
            from { transform: translate(-1%, -1%) scale(1); }
            to { transform: translate(2%, 2%) scale(1.06); }
        }

        @keyframes slowRotate {
            from { transform: rotate(0deg) scale(1); }
            to { transform: rotate(360deg) scale(1.1); }
        }

        @keyframes glowPulse {
            from { opacity: 0.5; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1.12); }
        }

        @keyframes floatOne {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(-9px, -17px); }
        }

        @keyframes floatTwo {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-13px); }
        }

        @keyframes sparkle {
            from { opacity: 0.2; transform: scale(0.65); }
            to { opacity: 1; transform: scale(1.5); }
        }

        @keyframes floatTitle {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }

        @keyframes gradientFlow {
            from { background-position: 0% center; }
            to { background-position: 240% center; }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(17px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes statusPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.45; }
        }

        @keyframes lineMove {
            from { left: 0; }
            to { left: 62%; }
        }

        @media (max-width: 600px) {
            .hero { padding: 30px 22px; min-height: 300px; }
            .orb-one { right: 5%; opacity: 0.65; }
            .description { font-size: 13px; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    </head>

    <body>
        <section class="hero" id="hero">
            <div class="ambient" id="ambient"></div>
            <div class="ambient-two" id="ambientTwo"></div>
            <div class="grid"></div>
            <div class="glow"></div>

            <div class="orb orb-one"></div>
            <div class="orb orb-two"></div>
            <div class="orb orb-three"></div>

            <div class="particle p1"></div>
            <div class="particle p2"></div>
            <div class="particle p3"></div>
            <div class="particle p4"></div>
            <div class="particle p5"></div>

            <div class="content">
                <div class="eyebrow">
                    <span class="status-dot"></span>
                    AI-POWERED CAREER EXPLORATION
                </div>

                <h1>
                    Discover your next<br>
                    <span class="gradient-text">career opportunity.</span>
                </h1>

                <p class="description">
                    Turn your resume into actionable career insights.
                    Explore matching roles, identify skill gaps, and
                    take the next step toward your professional future
                    with CareerVista AI.
                </p>
            </div>

            <div class="bottom-line"></div>
        </section>

        <script>
        (() => {
            const hero = document.getElementById("hero");
            const ambient = document.getElementById("ambient");
            const ambientTwo = document.getElementById("ambientTwo");

            let animationFrame = null;
            let currentX = 0;
            let currentY = 0;
            let targetX = 0;
            let targetY = 0;

            function animatePointer() {
                currentX += (targetX - currentX) * 0.09;
                currentY += (targetY - currentY) * 0.09;

                ambient.style.translate =
                    `${currentX * 1.2}px ${currentY * 1.2}px`;

                ambientTwo.style.translate =
                    `${currentX * -0.7}px ${currentY * -0.7}px`;

                if (
                    Math.abs(targetX - currentX) > 0.1 ||
                    Math.abs(targetY - currentY) > 0.1
                ) {
                    animationFrame = requestAnimationFrame(animatePointer);
                } else {
                    animationFrame = null;
                }
            }

            hero.addEventListener("pointermove", (event) => {
                const rect = hero.getBoundingClientRect();
                const x = (event.clientX - rect.left) / rect.width;
                const y = (event.clientY - rect.top) / rect.height;

                targetX = (x - 0.5) * 24;
                targetY = (y - 0.5) * 20;

                if (!animationFrame) {
                    animationFrame = requestAnimationFrame(animatePointer);
                }
            });

            function resetPointer() {
                targetX = 0;
                targetY = 0;

                if (!animationFrame) {
                    animationFrame = requestAnimationFrame(animatePointer);
                }
            }

            hero.addEventListener("pointerleave", resetPointer);
            hero.addEventListener("pointercancel", resetPointer);
        })();
        </script>
    </body>
    </html>
    """

    html = (
        html.replace("HERO_BACKGROUND", hero_background)
        .replace("HERO_TEXT", hero_text)
        .replace("HERO_SUBTEXT", hero_subtext)
        .replace("HERO_BORDER", hero_border)
    )

    components.html(html, height=305, scrolling=False)


show_hero(theme)

st.title("CareerVista AI")
st.caption("Resume-to-job matching for students and fresh graduates.")


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data(csv_path, modified_time):
    # Refresh the cache when the dataset changes.
    del modified_time

    data = pd.read_csv(csv_path).fillna("")

    required_columns = [
        "job_title",
        "category",
        "required_skills",
        "preferred_skills",
        "job_description",
    ]

    for column in required_columns:
        if column not in data.columns:
            data[column] = ""

    return data


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place "
        "'careervista_pk_starter_dataset.csv' in the same folder as app.py."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset contains no job records.")
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def split_skills(value):
    return [
        skill.strip()
        for skill in str(value).split(";")
        if skill.strip()
    ]


def safe_text(value, fallback="Not listed"):
    text = str(value).strip()
    return text if text else fallback


def extract_resume(uploaded):
    if uploaded is None:
        return ""

    filename = uploaded.name.lower()

    try:
        if filename.endswith(".txt"):
            return uploaded.getvalue().decode("utf-8", errors="ignore")

        if filename.endswith(".pdf"):
            if PdfReader is None:
                st.error("Install pypdf to enable PDF resume extraction.")
                return ""

            reader = PdfReader(uploaded)
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if filename.endswith(".docx"):
            if Document is None:
                st.error("Install python-docx to enable DOCX extraction.")
                return ""

            document = Document(uploaded)
            return "\n".join(
                paragraph.text for paragraph in document.paragraphs
            )

    except Exception as exc:
        st.error(f"Unable to read the resume: {exc}")

    return ""


def match_jobs(resume_text, data, top_n=5):
    resume_normalized = normalize(resume_text)

    if not resume_normalized or data.empty:
        return pd.DataFrame()

    job_texts = (
        data["job_title"].astype(str) + " " +
        data["category"].astype(str) + " " +
        data["job_description"].astype(str) + " " +
        data["required_skills"].astype(str) + " " +
        data["preferred_skills"].astype(str)
    ).tolist()

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(job_texts + [resume_normalized])
        similarities = cosine_similarity(
            matrix[-1], matrix[:-1]
        ).ravel()
    except ValueError:
        similarities = [0.0] * len(data)

    rows = []

    for index, (_, job) in enumerate(data.iterrows()):
        required = split_skills(job["required_skills"])

        matched = [
            skill for skill in required
            if normalize(skill) in resume_normalized
        ]

        missing = [
            skill for skill in required
            if normalize(skill) not in resume_normalized
        ]

        skill_score = len(matched) / max(len(required), 1)

        score = (
            0.75 * float(similarities[index])
            + 0.25 * skill_score
        ) * 100

        rows.append({
            "job_title": job["job_title"],
            "category": job["category"],
            "match_score": round(score, 1),
            "matched_skills": "; ".join(matched)
                or "No exact required-skill phrase found",
            "missing_skills": "; ".join(missing)
                or "No listed required skills missing",
            "experience_level": job.get("experience_level", ""),
            "education": job.get("education", ""),
            "job_description": job["job_description"],
            "required_skills": job["required_skills"],
            "location": job.get("country", "Pakistan"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("match_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def predict_category(resume_text, data):
    usable = data[
        data["category"].astype(str).str.strip() != ""
    ].copy()

    category_counts = usable["category"].value_counts()

    usable = usable[
        usable["category"].isin(
            category_counts[category_counts >= 2].index
        )
    ]

    if len(usable) < 4 or usable["category"].nunique() < 2:
        return None

    training_text = (
        usable["job_title"].astype(str) + " " +
        usable["job_description"].astype(str) + " " +
        usable["required_skills"].astype(str) + " " +
        usable["preferred_skills"].astype(str)
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            ),
        ),
        ("classifier", LinearSVC()),
    ])

    try:
        model.fit(training_text, usable["category"].astype(str))
        return model.predict([resume_text])[0]
    except ValueError:
        return None


# =========================================================
# RESUME DASHBOARD
# =========================================================
if page == "Resume Dashboard":
    st.subheader("Find roles that match your skills")

    st.write(
        "Upload your resume or paste your resume text below. "
        "Remove sensitive personal information before uploading."
    )

    uploaded = st.file_uploader(
        "Upload Resume (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="resume_upload",
    )

    resume_text = extract_resume(uploaded)

    pasted_text = st.text_area(
        "Or paste resume text here",
        height=170,
        placeholder="Skills, education, projects, internships...",
    )

    if pasted_text.strip():
        resume_text = pasted_text.strip()

    if st.button(
        "Analyze My Resume",
        type="primary",
        use_container_width=True,
    ):
        if len(resume_text.strip()) < 30:
            st.warning(
                "Please provide more resume text for useful matching."
            )
        else:
            with st.spinner("Analyzing your resume..."):
                st.session_state["last_resume"] = resume_text
                st.session_state["last_matches"] = match_jobs(
                    resume_text, df, 5
                )
                st.session_state["last_category"] = predict_category(
                    resume_text, df
                )

    if "last_matches" in st.session_state:
        matches = st.session_state["last_matches"]
        category = st.session_state.get("last_category")

        col1, col2, col3 = st.columns(3)

        col1.metric("Job Records", len(df))
        col2.metric("Categories", df["category"].nunique())

        top_score = (
            f"{matches.iloc[0]['match_score']:.1f}/100"
            if not matches.empty else "—"
        )
        col3.metric("Top Match Score", top_score)

        if category:
            st.success(f"Predicted Job Category: {category}")
        else:
            st.info(
                "Category prediction is unavailable for this dataset. "
                "Job matching can still work."
            )

        st.subheader("Your Top 5 Job Matches")

        if matches.empty:
            st.warning(
                "No matches could be calculated. Check your resume "
                "text and the dataset text fields."
            )
        else:
            for _, row in matches.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"### {row['job_title']} · "
                        f"{row['match_score']}% match"
                    )

                    st.caption(
                        f"{row['category']} | "
                        f"{safe_text(row['experience_level'])} | "
                        f"{safe_text(row['location'])}"
                    )

                    st.write(
                        safe_text(
                            row["job_description"],
                            "Description not listed",
                        )
                    )

                    st.write(
                        "**Required Skills:** "
                        + safe_text(row["required_skills"])
                    )

                    st.write(
                        "**Skills Found in Your Resume:** "
                        + row["matched_skills"]
                    )

                    st.write(
                        "**Skills to Consider Learning:** "
                        + row["missing_skills"]
                    )

            st.caption(
                "Match scores are ranking aids, not hiring probabilities "
                "or guarantees of employment."
            )

            st.download_button(
                "Download Recommendations CSV",
                data=matches.to_csv(index=False).encode("utf-8"),
                file_name="careervista_recommendations.csv",
                mime="text/csv",
            )

    st.divider()
    st.subheader("Dataset Overview")

    st.caption(
        "This chart shows records in the dataset, not live market demand."
    )

    counts = (
        df["category"]
        .replace("", "Uncategorized")
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Records")
    )

    fig = px.bar(
        counts,
        x="Category",
        y="Records",
        title="Job Records by Category",
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# EXPLORE JOBS
# =========================================================
elif page == "Explore Jobs":
    st.subheader("Explore Job Roles")

    categories = sorted(
        df["category"]
        .replace("", "Uncategorized")
        .astype(str)
        .unique()
        .tolist()
    )

    chosen_category = st.selectbox(
        "Filter by Category",
        ["All"] + categories,
    )

    keyword = st.text_input(
        "Search job titles, skills, or descriptions"
    )

    view = df.copy()

    if chosen_category != "All":
        if chosen_category == "Uncategorized":
            view = view[
                view["category"].astype(str).str.strip() == ""
            ]
        else:
            view = view[
                view["category"].astype(str) == chosen_category
            ]

    if keyword.strip():
        pattern = re.escape(keyword.strip())

        mask = view.astype(str).apply(
            lambda column: column.str.contains(
                pattern,
                case=False,
                na=False,
                regex=True,
            )
        ).any(axis=1)

        view = view[mask]

    st.write(f"Showing {len(view)} records")

    display_columns = [
        column for column in [
            "job_title",
            "category",
            "experience_level",
            "education",
            "required_skills",
            "preferred_skills",
            "job_description",
            "country",
        ]
        if column in view.columns
    ]

    st.dataframe(
        view[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Jobs",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_jobs.csv",
        mime="text/csv",
    )


# =========================================================
# JOB RECORDS
# =========================================================
elif page == "Job Records":
    st.subheader("Dataset Records")

    st.write("Search, review, and export the available job records.")

    query = st.text_input("Search All Record Fields")
    view = df.copy()

    if query.strip():
        pattern = re.escape(query.strip())

        mask = view.astype(str).apply(
            lambda column: column.str.contains(
                pattern,
                case=False,
                na=False,
                regex=True,
            )
        ).any(axis=1)

        view = view[mask]

    st.metric("Matching Records", len(view))

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Export Records as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="careervista_job_records.csv",
        mime="text/csv",
    )

    st.info(
        "Verify the dataset source and licensing before presenting "
        "these records as real or current vacancies."
    )


# =========================================================
# MODEL AND PROJECT INFORMATION
# =========================================================
else:
    st.subheader("Model & Project Information")

    st.markdown(
        """
        <div class="project-intro">
            <p>
                Welcome to CareerVista AI — a resume-to-job matching
                prototype designed to help students and fresh graduates
                explore career options, understand skill requirements,
                and identify areas for professional development.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="project-section-label">Project Overview</div>',
        unsafe_allow_html=True,
    )

    overview_cards = [
        (
            "🎯",
            "Problem Statement",
            "Students and fresh graduates can find it difficult to "
            "identify career roles that align with their skills and "
            "education. This project helps organize job information "
            "and discover relevant roles."
        ),
        (
            "🧠",
            "Machine Learning",
            "TF-IDF text features and a LinearSVC classifier are used "
            "for category prediction when the dataset contains enough "
            "repeated category labels."
        ),
        (
            "🔎",
            "Recommendation System",
            "Job recommendations are ranked using TF-IDF cosine "
            "similarity and required-skill phrase overlap to highlight "
            "potentially relevant opportunities."
        ),
        (
            "📝",
            "Text Preprocessing",
            "Missing text fields are handled, skill lists are separated "
            "using semicolons, and text is normalized to support "
            "consistent matching."
        ),
        (
            "📊",
            "Data Visualization",
            "Category charts summarize the job records available in "
            "the dataset and help users explore the distribution "
            "of listed roles."
        ),
        (
            "⚠️",
            "Limitations",
            "The starter dataset is intended for a prototype. It is not "
            "a live vacancy database, and reliable performance claims "
            "require a larger, independently evaluated dataset."
        ),
    ]

    card_columns = st.columns(2)

    for index, (icon, title, description) in enumerate(overview_cards):
        with card_columns[index % 2]:
            st.markdown(
                f"""
                <div class="project-card">
                    <div class="card-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="project-section-label">How the System Works</div>',
        unsafe_allow_html=True,
    )

    workflow_cards = [
        (
            "1",
            "Provide Resume",
            "Upload a PDF, DOCX, or TXT file, or paste resume text "
            "into the dashboard."
        ),
        (
            "2",
            "Analyze Text",
            "The system processes resume text and compares it with "
            "job titles, descriptions, and skills."
        ),
        (
            "3",
            "Rank Job Matches",
            "Similarity and required-skill overlap are combined to "
            "rank the most relevant records."
        ),
        (
            "4",
            "Review Results",
            "Explore suggested roles, skills found, and skills to "
            "consider learning next."
        ),
    ]

    workflow_columns = st.columns(2)

    for index, (number, title, description) in enumerate(workflow_cards):
        with workflow_columns[index % 2]:
            st.markdown(
                f"""
                <div class="project-card">
                    <div class="card-icon"> {number}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="project-section-label">Dataset Summary</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Dataset Rows", len(df))
    metric2.metric("Unique Categories", df["category"].nunique())
    metric3.metric(
        "Available Fields",
        len(df.columns),
    )

    st.caption(
        "CareerVista AI is an educational prototype. Match scores are "
        "ranking aids, not hiring probabilities or guarantees of employment."
    )
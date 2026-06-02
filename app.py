"""
Student Academic Performance Predictor
Run: streamlit run app.py
Requires: pip install streamlit pandas numpy scikit-learn xgboost plotly joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config 
st.set_page_config(
    page_title="EduPredict — Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0f1e35 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #90cdf4, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #a0aec0;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #63b3ed;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.4), transparent);
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #111827, #1a2233);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #0f1c2e, #152236);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,179,237,0.35);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #718096;
}

/* ── Result Banner ── */
.result-pass {
    background: linear-gradient(135deg, #0d2618, #1a3a28);
    border: 1.5px solid #48bb78;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}
.result-fail {
    background: linear-gradient(135deg, #2d1515, #3a1a1a);
    border: 1.5px solid #fc8181;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-subtitle {
    font-size: 0.9rem;
    color: #a0aec0;
    font-weight: 300;
}

/* ── Risk Meter ── */
.risk-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 12px;
    overflow: hidden;
    margin: 0.6rem 0;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}

/* ── Factor Pill ── */
.factor-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.factor-positive { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.factor-negative { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }
.factor-neutral  { background: rgba(99,179,237,0.12); color: #63b3ed; border: 1px solid rgba(99,179,237,0.25); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1422 !important;
    border-right: 1px solid rgba(99,179,237,0.08) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(49,130,206,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #4299e1) !important;
    box-shadow: 0 6px 28px rgba(49,130,206,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #111827 !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.1) !important; }

/* ── Tooltip box ── */
.info-box {
    background: rgba(99,179,237,0.07);
    border-left: 3px solid #63b3ed;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #90cdf4;
    margin: 0.5rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

/* ── Recommendation Cards ── */
.rec-card {
    background: linear-gradient(135deg, #0f1c2e, #152236);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s, transform 0.2s;
}
.rec-card:hover {
    border-color: rgba(99,179,237,0.38);
    transform: translateY(-2px);
}
.rec-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}
.rec-icon {
    font-size: 1.5rem;
    line-height: 1;
}
.rec-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: #e2e8f0;
}
.rec-priority {
    margin-left: auto;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
}
.priority-critical { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3);}
.priority-high     { background: rgba(246,173,85,0.15);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.3);}
.priority-medium   { background: rgba(99,179,237,0.12);  color: #63b3ed; border: 1px solid rgba(99,179,237,0.25);}
.priority-low      { background: rgba(72,187,120,0.12);  color: #68d391; border: 1px solid rgba(72,187,120,0.25);}
.rec-body {
    font-size: 0.84rem;
    color: #a0aec0;
    line-height: 1.6;
}
.rec-action {
    margin-top: 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #63b3ed;
    display: flex;
    align-items: center;
    gap: 5px;
}
.rec-category-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 8px;
    margin-bottom: 0.8rem;
}
.cat-academic   { background: rgba(159,122,234,0.15); color: #b794f4; border: 1px solid rgba(159,122,234,0.25);}
.cat-engagement { background: rgba(99,179,237,0.12);  color: #63b3ed; border: 1px solid rgba(99,179,237,0.25);}
.cat-support    { background: rgba(246,173,85,0.12);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.25);}
.cat-admin      { background: rgba(72,187,120,0.12);  color: #68d391; border: 1px solid rgba(72,187,120,0.25);}
.cat-pastoral   { background: rgba(252,129,129,0.12); color: #fc8181; border: 1px solid rgba(252,129,129,0.25);}
.peer-card {
    background: linear-gradient(135deg, #0a1628, #0f1e35);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.7rem;
}
.peer-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.83rem;
    color: #718096;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.peer-stat:last-child { border-bottom: none; }
.peer-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
}
.milestone-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(99,179,237,0.06);
    font-size: 0.84rem;
}
.milestone-row:last-child { border-bottom: none; }
.milestone-check {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    flex-shrink: 0;
}
.m-done { background: rgba(72,187,120,0.2); color: #68d391; }
.m-warn { background: rgba(246,173,85,0.2); color: #f6ad55; }
.m-todo { background: rgba(252,129,129,0.2); color: #fc8181; }
</style>
""", unsafe_allow_html=True)


# ── Helper: Build feature vector from user inputs 
def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """Convert UI inputs → engineered feature vector matching training schema."""

    edu_map  = {"No Formal Quals": 0, "Lower Than A Level": 0.25,
                 "A Level": 0.5, "HE Qualification": 0.75, "Post Graduate": 1.0}
    age_map  = {"0–35": 0, "35–55": 1, "55+": 2}
    imd_map  = {"0–10% (Most Deprived)": 1/9, "10–20%": 2/9, "20–30%": 3/9,
                 "30–40%": 4/9, "40–50%": 5/9, "50–60%": 6/9,
                 "60–70%": 7/9, "70–80%": 8/9, "80–90%+": 1.0}
    mod_map  = {"AAA":0,"BBB":1,"CCC":2,"DDD":3,"EEE":4,"FFF":5,"GGG":6}
    reg_map  = {"East Anglian":0,"East Midlands":1,"Ireland":2,"London":3,
                 "North":4,"North Western":5,"Scotland":6,"South East":7,
                 "South West":8,"Wales":9,"West Midlands":10,"Yorkshire":11}
    atype_map = {"CMA": 0, "Exam": 1, "TMA": 2}

    score          = inputs["score"]
    weight         = inputs["weight"]
    sum_click      = inputs["sum_click"]
    studied_credits= inputs["studied_credits"]
    mod_len        = inputs["module_presentation_length"]
    date_reg       = inputs["date_registration"]
    date_sub       = inputs["date_submitted"]
    date_assess    = inputs["date_assessment"]
    prev_attempts  = inputs["num_of_prev_attempts"]

    weighted_score          = score * weight / 100
    enrollment_duration     = max(mod_len - date_reg, 0)
    submission_timeliness   = date_assess - date_sub
    module_completion_ratio = min(date_assess / max(mod_len, 1), 1.0)
    click_density           = sum_click / max(enrollment_duration, 1)
    study_efficiency        = weighted_score / max(sum_click, 1)
    early_registration      = 1 if date_reg < 0 else 0
    unregistered            = inputs["unregistered"]
    high_engager            = 1 if sum_click > 1500 else 0
    repeat_attempter        = 1 if prev_attempts > 0 else 0
    heavy_course_load       = 1 if studied_credits > 120 else 0
    score_tier              = 0 if score < 40 else (1 if score < 55 else (2 if score < 70 else 3))

    # Student-level estimates (single assessment — use score as proxy)
    student_avg_score           = score
    student_score_std           = 0.0
    student_total_clicks        = sum_click
    student_num_assessments     = 1
    student_avg_weighted_score  = weighted_score
    student_avg_timeliness      = submission_timeliness

    # Module context (use neutral defaults if no model loaded)
    module_avg_score        = 65.0
    module_pass_rate        = 0.65
    module_avg_clicks       = 0.42
    relative_score_in_module= (score - module_avg_score) / max(module_avg_score, 1)

    row = {
        "education_level"           : edu_map.get(inputs["highest_education"], 0.75),
        "age_numeric"               : age_map.get(inputs["age_band"], 1),
        "imd_score"                 : imd_map.get(inputs["imd_band"], 0.5),
        "gender_binary"             : 1 if inputs["gender"] == "Male" else 0,
        "disability_flag"           : 1 if inputs["disability"] == "Yes" else 0,
        "code_module_encoded"       : mod_map.get(inputs["code_module"], 0),
        "region_encoded"            : reg_map.get(inputs["region"], 0),
        "assessment_type_encoded"   : atype_map.get(inputs["assessment_type"], 2),
        "early_registration"        : early_registration,
        "unregistered"              : unregistered,
        "enrollment_duration"       : enrollment_duration,
        "submission_timeliness"     : submission_timeliness,
        "module_completion_ratio"   : module_completion_ratio,
        "weighted_score"            : weighted_score,
        "score_tier"                : score_tier,
        "click_density"             : click_density,
        "study_efficiency"          : study_efficiency,
        "high_engager"              : high_engager,
        "repeat_attempter"          : repeat_attempter,
        "heavy_course_load"         : heavy_course_load,
        "student_avg_score"         : student_avg_score,
        "student_score_std"         : student_score_std,
        "student_total_clicks"      : student_total_clicks,
        "student_num_assessments"   : student_num_assessments,
        "student_avg_weighted_score": student_avg_weighted_score,
        "student_avg_timeliness"    : student_avg_timeliness,
        "module_avg_score"          : module_avg_score,
        "module_pass_rate"          : module_pass_rate,
        "module_avg_clicks"         : module_avg_clicks,
        "relative_score_in_module"  : relative_score_in_module,
        "num_of_prev_attempts"      : prev_attempts,
        "studied_credits"           : studied_credits,
        "module_presentation_length": mod_len,
    }
    return pd.DataFrame([row])


def heuristic_predict(fv: pd.DataFrame) -> tuple[float, str]:
    """Rule-based prediction when no model file is found."""
    row   = fv.iloc[0]
    score = 0.0
    score += row["weighted_score"] / 100 * 0.35
    score += row["study_efficiency"] * 15 * 0.20
    score += row["click_density"] / 30 * 0.15
    score += row["education_level"] * 0.10
    score += (1 - row["imd_score"]) * 0.05
    score += row["module_completion_ratio"] * 0.08
    score -= row["repeat_attempter"] * 0.07
    score += row["submission_timeliness"] / 50 * 0.05
    prob = float(np.clip(score, 0.02, 0.98))
    label = "Pass" if prob >= 0.5 else "At-Risk"
    return prob, label


def load_model():
    for path in ["model_xgboost.pkl", "model_random_forest.pkl",
                 "model_logistic_regression.pkl"]:
        if os.path.exists(path):
            return joblib.load(path), path.replace(".pkl","").replace("model_","").replace("_"," ").title()
    return None, "Heuristic Engine"


def gauge_chart(prob: float) -> go.Figure:
    pct   = prob * 100
    color = "#48bb78" if prob >= 0.6 else ("#f6ad55" if prob >= 0.4 else "#fc8181")
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        number= {"suffix": "%", "font": {"size": 38, "color": color,
                                          "family": "Syne"}},
        gauge = {
            "axis"    : {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#4a5568", "tickfont": {"color":"#718096","size":10}},
            "bar"     : {"color": color, "thickness": 0.28},
            "bgcolor" : "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps"   : [
                {"range": [0,  40], "color": "rgba(252,129,129,0.08)"},
                {"range": [40, 60], "color": "rgba(246,173,85,0.08)"},
                {"range": [60,100], "color": "rgba(72,187,120,0.08)"},
            ],
            "threshold": {"line":{"color":"white","width":2},"thickness":0.75,"value":50}
        },
        title={"text": "Pass Probability", "font":{"size":13,"color":"#718096","family":"DM Sans"}}
    ))
    fig.update_layout(
        height=230, margin=dict(t=20,b=0,l=20,r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0"
    )
    return fig


def radar_chart(fv: pd.DataFrame) -> go.Figure:
    row = fv.iloc[0]
    cats = ["Score","Engagement","Efficiency","Timeliness","Education","Completion"]
    vals = [
        min(row["weighted_score"] / 100, 1.0),
        min(row["click_density"]  / 30,  1.0),
        min(row["study_efficiency"] * 15, 1.0),
        min(max((row["submission_timeliness"] + 30) / 60, 0), 1.0),
        row["education_level"],
        row["module_completion_ratio"],
    ]
    vals_pct = [v * 100 for v in vals]

    fig = go.Figure(go.Scatterpolar(
        r    = vals_pct + [vals_pct[0]],
        theta= cats + [cats[0]],
        fill = "toself",
        fillcolor= "rgba(99,179,237,0.12)",
        line = dict(color="#63b3ed", width=2),
        name = "Student Profile"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(color="#4a5568",size=9),
                            gridcolor="rgba(99,179,237,0.08)"),
            angularaxis=dict(tickfont=dict(color="#90cdf4",size=11),
                             gridcolor="rgba(99,179,237,0.08)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=30,b=30,l=40,r=40),
        height=300,
        showlegend=False,
        font_color="#e8eaf0"
    )
    return fig


def factor_bar(fv: pd.DataFrame) -> go.Figure:
    row    = fv.iloc[0]
    labels = ["Assessment Score","Online Engagement","Study Efficiency",
              "Submission Timeliness","Education Level","Course Completion",
              "Prior Attempts (-)","Credits Load"]
    scores = [
        row["weighted_score"] / 100,
        min(row["click_density"] / 30, 1),
        min(row["study_efficiency"] * 15, 1),
        min(max((row["submission_timeliness"]+30)/60, 0), 1),
        row["education_level"],
        row["module_completion_ratio"],
        1 - row["repeat_attempter"] * 0.8,
        1 - row["heavy_course_load"] * 0.4,
    ]
    colors = ["#48bb78" if s >= 0.6 else ("#f6ad55" if s >= 0.35 else "#fc8181")
              for s in scores]

    fig = go.Figure(go.Bar(
        x          = [s * 100 for s in scores],
        y          = labels,
        orientation= "h",
        marker_color    = colors,
        marker_line_width=0,
        text       = [f"{s*100:.0f}%" for s in scores],
        textposition= "outside",
        textfont   = dict(color="#a0aec0", size=10)
    ))
    fig.update_layout(
        xaxis=dict(range=[0,115], showgrid=False, zeroline=False,
                   tickfont=dict(color="#4a5568"), showticklabels=False),
        yaxis=dict(tickfont=dict(color="#a0aec0", size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=10,b=10,l=10,r=60),
        height=300,
        bargap=0.35,
        font_color="#e8eaf0"
    )
    return fig


def score_trend_chart(score, weight) -> go.Figure:
    """Simulate how weighted score contributes across common weight scenarios."""
    weights = [5, 10, 15, 20, 25, 30, 35, 40]
    ws      = [score * w / 100 for w in weights]
    highlight = score * weight / 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weights, y=ws, mode="lines+markers",
        line=dict(color="#63b3ed", width=2),
        marker=dict(size=6, color="#63b3ed"),
        fill="tozeroy", fillcolor="rgba(99,179,237,0.07)",
        name="Weighted Score"
    ))
    fig.add_vline(x=weight, line_dash="dot", line_color="#f6ad55", line_width=1.5)
    fig.add_annotation(x=weight, y=highlight,
                       text=f"  Current: {highlight:.1f}",
                       showarrow=False, font=dict(color="#f6ad55", size=11))
    fig.update_layout(
        xaxis=dict(title="Assessment Weight (%)", tickfont=dict(color="#718096",size=10),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Weighted Score", tickfont=dict(color="#718096",size=10),
                   gridcolor="rgba(255,255,255,0.04)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=10,b=40,l=50,r=20),
        height=220,
        showlegend=False,
        font_color="#e8eaf0"
    )
    return fig


# ── Recommendation Engine ─────────────────────────────────────────────────────

def generate_recommendations(fv: pd.DataFrame, inputs: dict, prob: float) -> list[dict]:
    """
    Rule-based recommendation engine.
    Returns a list of recommendation dicts ordered by priority weight.
    Each dict: {title, body, action, icon, priority, category}
    """
    row  = fv.iloc[0]
    recs = []

    score        = inputs["score"]
    sum_click    = inputs["sum_click"]
    weight       = inputs["weight"]
    prev         = inputs["num_of_prev_attempts"]
    date_sub     = inputs["date_submitted"]
    date_assess  = inputs["date_assessment"]
    credits      = inputs["studied_credits"]
    imd          = row["imd_score"]
    edu          = row["education_level"]
    unreg        = inputs["unregistered"]
    timeliness   = row["submission_timeliness"]
    click_d      = row["click_density"]
    mod_comp     = row["module_completion_ratio"]

    # ── ACADEMIC ──────────────────────────────────────────────────────────────
    if score < 40:
        recs.append(dict(
            title    = "Urgent: Score Below Pass Threshold",
            body     = (f"This assessment score of {score} is below the 40-point pass mark. "
                        "Without intervention, this student is at immediate risk of failing "
                        "the module. A one-to-one tutorial with the module tutor should be "
                        "arranged as soon as possible."),
            action   = "Book emergency tutorial · Review marking scheme · Check for extenuating circumstances",
            icon     = "🚨", priority = "critical", category = "academic", weight = 10
        ))
    elif score < 55:
        recs.append(dict(
            title    = "Score Below Merit Band — Targeted Support Needed",
            body     = (f"Score of {score} passes but sits below the merit band (55). "
                        "Supplementary reading and practice questions on weak topics "
                        "will help the student progress to stronger performance bands."),
            action   = "Share topic revision packs · Suggest practice quizzes · Set a follow-up check-in",
            icon     = "📉", priority = "high", category = "academic", weight = 7
        ))
    elif score >= 70:
        recs.append(dict(
            title    = "Strong Score — Encourage Stretch Goals",
            body     = (f"Assessment score of {score} is excellent. Channel this momentum "
                        "into harder problem sets or optional enrichment material to maintain "
                        "motivation throughout the rest of the module."),
            action   = "Share extension resources · Nominate for peer mentoring programme",
            icon     = "🏆", priority = "low", category = "academic", weight = 1
        ))

    if weight >= 40 and score < 55:
        recs.append(dict(
            title    = "High-Weight Assessment at Risk",
            body     = (f"This assessment carries {weight}% of the module grade. "
                        "A score of {score} on a high-weight item has a disproportionate "
                        "impact on final result. Consider whether a resit or late "
                        "submission policy applies."),
            action   = "Check resit eligibility · Escalate to academic advisor",
            icon     = "⚖️", priority = "critical", category = "academic", weight = 9
        ))

    # ── ENGAGEMENT ────────────────────────────────────────────────────────────
    if click_d < 2:
        recs.append(dict(
            title    = "Very Low VLE Engagement Detected",
            body     = (f"Click density of {click_d:.1f} clicks/day is critically low. "
                        "Students with this level of VLE activity are at high risk of "
                        "silent withdrawal. An outreach email or phone call should be made "
                        "within 48 hours."),
            action   = "Send engagement nudge email · Flag to personal tutor · Check login logs",
            icon     = "📵", priority = "critical", category = "engagement", weight = 9
        ))
    elif click_d < 5:
        recs.append(dict(
            title    = "Below-Average Online Activity",
            body     = (f"Click density ({click_d:.1f}/day) is below the module average. "
                        "Encourage the student to use VLE resources more consistently — "
                        "forums, lecture recordings and formative quizzes are strong "
                        "predictors of success."),
            action   = "Send VLE guide · Highlight key resources in module · Set engagement goal",
            icon     = "💻", priority = "high", category = "engagement", weight = 6
        ))
    elif click_d >= 15:
        recs.append(dict(
            title    = "High Engagement — Maintain Momentum",
            body     = (f"Strong VLE engagement ({click_d:.1f} clicks/day). Recognise this "
                        "positive behaviour and ensure the student is directing time to "
                        "high-value activities rather than passive browsing."),
            action   = "Positive reinforcement message · Suggest active learning resources",
            icon     = "🔥", priority = "low", category = "engagement", weight = 1
        ))

    # ── SUBMISSION TIMELINESS ─────────────────────────────────────────────────
    if timeliness < 0:
        recs.append(dict(
            title    = "Late Submission — Investigate Causes",
            body     = (f"Submission was {abs(int(timeliness))} day(s) late. "
                        "Late submissions often indicate competing pressures (work, "
                        "caring responsibilities, mental health). A welfare check is "
                        "recommended alongside a discussion about time management."),
            action   = "Log late submission · Offer time-management coaching · Check wellbeing",
            icon     = "⏰", priority = "high", category = "support", weight = 7
        ))
    elif timeliness < 2:
        recs.append(dict(
            title    = "Submitted Very Close to Deadline",
            body     = ("Submission made within 2 days of the due date. While not late, "
                        "this pattern can indicate poor planning or last-minute cramming. "
                        "Discuss the student's study schedule in the next meeting."),
            action   = "Review study plan · Suggest weekly planning techniques",
            icon     = "⏱️", priority = "medium", category = "support", weight = 4
        ))

    # ── PREVIOUS ATTEMPTS ─────────────────────────────────────────────────────
    if prev >= 2:
        recs.append(dict(
            title    = f"Multiple Previous Attempts ({prev}x) — Root Cause Review",
            body     = (f"This student has attempted the module {prev} times before. "
                        "Recurring attempts without intervention usually signal an "
                        "unresolved barrier — learning difficulty, unsuitable study "
                        "method, or external life factors. A structured review meeting "
                        "is strongly advised."),
            action   = "Schedule student review meeting · Access previous attempt records · Refer to study skills team",
            icon     = "🔄", priority = "critical", category = "pastoral", weight = 8
        ))
    elif prev == 1:
        recs.append(dict(
            title    = "First Re-attempt — Monitoring Required",
            body     = ("Student is on their first re-attempt. Ensure they have reflected "
                        "on what went wrong previously and have an updated study plan in "
                        "place. Monitor engagement closely over the next 4 weeks."),
            action   = "Compare with previous attempt data · Agree a milestone plan",
            icon     = "🔁", priority = "medium", category = "pastoral", weight = 5
        ))

    # ── DEPRIVATION / FINANCIAL ───────────────────────────────────────────────
    if imd < 0.30:
        recs.append(dict(
            title    = "High Deprivation Area — Financial & Pastoral Support",
            body     = ("Student is registered from a highly deprived area (bottom 30% IMD). "
                        "Students in this band are statistically more likely to experience "
                        "financial stress, housing instability and health issues that affect "
                        "study. Proactively signpost available bursaries and pastoral services."),
            action   = "Share hardship fund info · Signpost food bank / housing services · Allocate extra pastoral check-ins",
            icon     = "🏘️", priority = "high", category = "support", weight = 6
        ))
    elif imd < 0.50:
        recs.append(dict(
            title    = "Moderate Deprivation — Awareness Flag",
            body     = ("Student is from a moderately deprived area. Worth checking whether "
                        "financial or logistical pressures are impacting study time, especially "
                        "if other risk signals are also present."),
            action   = "Include in next pastoral contact · Share bursary information",
            icon     = "🏠", priority = "medium", category = "support", weight = 3
        ))

    # ── UNREGISTRATION ────────────────────────────────────────────────────────
    if unreg:
        recs.append(dict(
            title    = "Student Has Unregistered Mid-Module — Immediate Action",
            body     = ("Unregistration is the strongest single predictor of dropout in the "
                        "OULAD dataset. If not already done, make direct contact with the "
                        "student to understand their situation. Re-registration or a "
                        "planned withdrawal is far preferable to a silent dropout."),
            action   = "Call / email student immediately · Explore re-registration window · Log welfare concern",
            icon     = "🚪", priority = "critical", category = "admin", weight = 10
        ))

    # ── CREDITS LOAD ──────────────────────────────────────────────────────────
    if credits > 180:
        recs.append(dict(
            title    = "Very Heavy Credit Load",
            body     = (f"Student is studying {credits} credits — significantly above the "
                        "recommended 120. This level of workload increases fatigue and "
                        "the probability of failing one or more modules. Discuss whether "
                        "a lighter load is feasible for the next presentation."),
            action   = "Review overall credit load · Discuss deferral of one module · Monitor stress signals",
            icon     = "📚", priority = "high", category = "admin", weight = 6
        ))
    elif credits > 120:
        recs.append(dict(
            title    = "Above-Average Credits — Monitor Workload",
            body     = (f"At {credits} credits, this student is carrying a heavier-than-average "
                        "workload. Keep an eye on engagement and performance trends across "
                        "all concurrent modules."),
            action   = "Cross-check performance in other modules · Advise on workload balance",
            icon     = "📖", priority = "medium", category = "admin", weight = 3
        ))

    # ── EDUCATION LEVEL ───────────────────────────────────────────────────────
    if edu <= 0.25 and prob < 0.6:
        recs.append(dict(
            title    = "Lower Prior Education — Academic Skills Support",
            body     = ("Students entering with below A-level qualifications may struggle "
                        "with academic writing, referencing and higher-order analysis. "
                        "Signposting to the Study Skills Centre early can prevent "
                        "avoidable failures."),
            action   = "Refer to Study Skills Centre · Share academic writing guides · Consider foundation reading list",
            icon     = "🎓", priority = "medium", category = "academic", weight = 4
        ))

    # ── OVERALL PROBABILITY ───────────────────────────────────────────────────
    if prob < 0.35:
        recs.append(dict(
            title    = "Overall High Dropout Risk — Escalate to Advisor",
            body     = (f"Pass probability of {prob*100:.0f}% places this student in the "
                        "high-risk tier. A combination of multiple risk factors is present. "
                        "This case should be escalated to the student's academic advisor "
                        "for a structured support plan."),
            action   = "Raise formal support plan · Notify academic advisor · Set 2-week review",
            icon     = "🆘", priority = "critical", category = "pastoral", weight = 10
        ))
    elif prob >= 0.75:
        recs.append(dict(
            title    = "On Track for a Pass — Nurture Continued Success",
            body     = (f"Strong pass probability ({prob*100:.0f}%). The student is performing "
                        "well across most indicators. Keep communication channels open and "
                        "look for opportunities to challenge and stretch this learner."),
            action   = "Recognition email · Offer advanced optional material",
            icon     = "🌟", priority = "low", category = "academic", weight = 1
        ))

    # Sort by weight descending, then remove the internal weight key
    recs.sort(key=lambda r: r["weight"], reverse=True)
    for r in recs:
        r.pop("weight")
    return recs


def peer_comparison(fv: pd.DataFrame, df_all: pd.DataFrame, module: str) -> dict:
    """Compare student's key metrics against module peers from dataset."""
    mod_df = df_all[df_all["code_module"] == module]
    if mod_df.empty:
        mod_df = df_all
    result = {}
    for col, label in [("score","Score"), ("sum_click","VLE Clicks"), ("studied_credits","Credits")]:
        if col in mod_df.columns:
            vals = pd.to_numeric(mod_df[col], errors="coerce").dropna()
            student_val = float(fv.iloc[0].get(
                col if col != "sum_click" else "student_total_clicks",
                fv.iloc[0].get(col, 0)
            ))
            pct = float((vals < student_val).mean() * 100)
            result[label] = {"student": student_val, "module_avg": float(vals.mean()),
                             "module_med": float(vals.median()), "percentile": pct}
    return result


@st.cache_data(show_spinner=False)
def load_student_data():
    paths = ["cleaned_student_data.csv",
             "/mnt/user-data/uploads/cleaned_student_data.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()


# SIDEBAR — Student Input Form


with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                    color:#63b3ed;letter-spacing:-0.3px;'>⚡ EduPredict</div>
        <div style='font-size:0.78rem;color:#4a5568;margin-top:2px;'>
            Student Performance AI
        </div>
    </div>
    <hr style='margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**👤 Student Demographics**")
    gender    = st.selectbox("Gender", ["Male", "Female"])
    age_band  = st.selectbox("Age Band", ["0–35", "35–55", "55+"])
    region    = st.selectbox("Region", ["East Anglian","East Midlands","Ireland",
                                         "London","North","North Western","Scotland",
                                         "South East","South West","Wales",
                                         "West Midlands","Yorkshire"])
    highest_education = st.selectbox("Highest Education",
                            ["No Formal Quals","Lower Than A Level","A Level",
                             "HE Qualification","Post Graduate"])
    imd_band  = st.selectbox("IMD Band (Deprivation)",
                    ["0–10% (Most Deprived)","10–20%","20–30%","30–40%","40–50%",
                     "50–60%","60–70%","70–80%","80–90%+"])
    disability = st.selectbox("Disability", ["No","Yes"])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("** Module & Assessment**")
    code_module    = st.selectbox("Module Code", ["AAA","BBB","CCC","DDD","EEE","FFF","GGG"])
    assessment_type= st.selectbox("Assessment Type", ["TMA","CMA","Exam"])
    score          = st.slider("Assessment Score", 0, 100, 72, help="Raw score on this assessment")
    weight         = st.slider("Assessment Weight (%)", 5, 100, 20, step=5)
    date_submitted = st.number_input("Date Submitted (days from start)", 0, 400, 50)
    date_assessment= st.number_input("Assessment Due Date (days from start)", 0, 400, 54)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("** Online Activity & Enrolment**")
    sum_click      = st.number_input("Total VLE Clicks", 0, 20000, 1200, step=100)
    studied_credits= st.number_input("Studied Credits", 0, 600, 120, step=30)
    num_of_prev_attempts = st.number_input("Previous Attempts", 0, 6, 0)
    module_presentation_length = st.number_input("Module Length (days)", 100, 400, 268)
    date_registration = st.number_input("Registration Day (negative = early)",
                                         -100, 200, -10)
    unregistered   = st.selectbox("Unregistered Mid-Module?", ["No","Yes"])
    unregistered   = 1 if unregistered == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("  Predict Performance")


# MAIN PANEL

st.markdown("""
<div class="hero">
    <div class="badge">AI-Powered · OULAD Framework</div>
    <h1> Student Performance Predictor</h1>
    <p>Enter student information in the sidebar and click <b>Predict Performance</b> to
       generate a detailed academic risk assessment with feature-level explanations.</p>
</div>
""", unsafe_allow_html=True)

# ── Before prediction: show guide
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📝", "Fill the Form",
         "Enter student demographics, module details, assessment scores and online activity in the sidebar."),
        (col2, "🤖", "AI Analysis",
         "Our model analyzes 33 engineered features including engagement patterns, submission timeliness and peer comparisons."),
        (col3, "📊", "Get Insights",
         "Receive pass probability, risk factors, radar profile and actionable recommendations."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;min-height:160px;">
                <div style="font-size:2.2rem;margin-bottom:0.7rem;">{icon}</div>
                <div style="font-family:Syne,sans-serif;font-size:1rem;
                            font-weight:700;color:#63b3ed;margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.85rem;color:#718096;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>Tip:</b> Load your trained model files 
        (<code>model_xgboost.pkl</code>, <code>model_random_forest.pkl</code>, or 
        <code>model_logistic_regression.pkl</code>) in the same directory as 
        <code>app.py</code> for ML-based predictions. Without them, a heuristic 
        engine is used automatically.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# PREDICTION LOGIC

inputs = dict(
    gender=gender, age_band=age_band, region=region,
    highest_education=highest_education, imd_band=imd_band, disability=disability,
    code_module=code_module, assessment_type=assessment_type,
    score=score, weight=weight,
    date_submitted=date_submitted, date_assessment=date_assessment,
    sum_click=sum_click, studied_credits=studied_credits,
    num_of_prev_attempts=num_of_prev_attempts,
    module_presentation_length=module_presentation_length,
    date_registration=date_registration, unregistered=unregistered,
)

fv           = build_feature_vector(inputs)
model, mname = load_model()

with st.spinner(" Analysing student profile..."):
    if model is not None:
        try:
            prob  = float(model.predict_proba(fv)[0][1])
            label = "Pass" if prob >= 0.5 else "At-Risk"
        except Exception:
            prob, label = heuristic_predict(fv)
            mname = "Heuristic Engine"
    else:
        prob, label = heuristic_predict(fv)


# RESULTS DISPLAY  (tabbed)

risk_pct   = prob * 100
is_pass    = prob >= 0.5
res_class  = "result-pass" if is_pass else "result-fail"
res_color  = "#48bb78"     if is_pass else "#fc8181"
res_emoji  = "✅"          if is_pass else "⚠️"
res_text   = "Likely to Pass" if is_pass else "At Risk"
confidence = "High" if abs(prob - 0.5) > 0.25 else ("Moderate" if abs(prob - 0.5) > 0.10 else "Low")

tab_pred, tab_rec = st.tabs(["📊  Prediction & Analysis", "💡  Recommendations"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:

    # ── Result Banner
    st.markdown(f"""
    <div class="{res_class}">
        <div class="result-title" style="color:{res_color};">{res_emoji} {res_text}</div>
        <div class="result-subtitle">
            Pass Probability: <b style="color:{res_color};">{risk_pct:.1f}%</b> &nbsp;|&nbsp;
            Confidence: <b>{confidence}</b> &nbsp;|&nbsp;
            Engine: <b>{mname}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Gauge + Radar + Factor Bar
    col_g, col_r, col_f = st.columns([1, 1.1, 1.2])

    with col_g:
        st.markdown('<div class="section-title">Pass Probability</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
        bar_color = "#48bb78" if is_pass else "#fc8181"
        st.markdown(f"""
        <div style="margin-top:-0.5rem;">
            <div style="display:flex;justify-content:space-between;
                        font-size:0.75rem;color:#718096;margin-bottom:4px;">
                <span>At Risk</span><span>Pass</span>
            </div>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill"
                     style="width:{risk_pct}%;background:{bar_color};"></div>
            </div>
            <div style="text-align:center;font-size:0.78rem;color:#4a5568;margin-top:4px;">
                Threshold at 50%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(fv), use_container_width=True, config={"displayModeBar": False})

    with col_f:
        st.markdown('<div class="section-title">Factor Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(factor_bar(fv), use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Quick Metrics
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    row = fv.iloc[0]
    metrics = [
        ("Weighted Score",    f"{row['weighted_score']:.1f}",      "/100",     "#63b3ed"),
        ("Click Density",     f"{row['click_density']:.2f}",        "/day",    "#9f7aea"),
        ("Study Efficiency",  f"{row['study_efficiency']*100:.1f}", "pts/kclk","#f6ad55"),
        ("Submission Lead",   f"{int(row['submission_timeliness'])}","days",    "#48bb78"),
        ("Module Completion", f"{row['module_completion_ratio']*100:.0f}","%", "#4fd1c5"),
        ("Engagement Level",  "High" if row['high_engager'] else "Normal","",  "#fc8181"),
    ]
    cols = st.columns(6)
    for col, (label, val, unit, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}<span style="font-size:0.9rem;color:#4a5568;">{unit}</span></div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 3: Score Trend + Intervention Tips
    st.markdown("<br>", unsafe_allow_html=True)
    col_t, col_i = st.columns([1.3, 1])

    with col_t:
        st.markdown('<div class="section-title">Weighted Score Across Weights</div>', unsafe_allow_html=True)
        st.plotly_chart(score_trend_chart(score, weight), use_container_width=True,
                        config={"displayModeBar": False})

    with col_i:
        st.markdown('<div class="section-title">Intervention Signals</div>', unsafe_allow_html=True)
        tips = []
        if score < 40:
            tips.append(("🔴", "Critical: Score below pass threshold (40). Immediate academic support needed.", "negative"))
        elif score < 55:
            tips.append(("🟡", "Score below merit threshold. Targeted tutoring recommended.", "neutral"))
        else:
            tips.append(("🟢", "Assessment score is healthy.", "positive"))
        if row["click_density"] < 3:
            tips.append(("🔴", "Very low online engagement. Student may be disengaged.", "negative"))
        elif row["click_density"] < 8:
            tips.append(("🟡", "Moderate engagement. Encourage regular VLE use.", "neutral"))
        else:
            tips.append(("🟢", "Strong online engagement detected.", "positive"))
        if row["submission_timeliness"] < 0:
            tips.append(("🔴", "Late submission. Discuss time management strategies.", "negative"))
        elif row["submission_timeliness"] < 3:
            tips.append(("🟡", "Submitted close to deadline. Monitor future submissions.", "neutral"))
        else:
            tips.append(("🟢", "Early submission — good time management.", "positive"))
        if row["repeat_attempter"]:
            tips.append(("🟡", f"{int(inputs['num_of_prev_attempts'])} prior attempt(s). Investigate recurring barriers.", "neutral"))
        if row["imd_score"] < 0.35:
            tips.append(("🟡", "High deprivation area. Consider financial/pastoral support.", "neutral"))
        if row["unregistered"]:
            tips.append(("🔴", "Student has unregistered. Risk of dropout is high.", "negative"))
        if not tips:
            tips.append(("🟢", "All indicators within healthy ranges.", "positive"))

        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        for emoji, msg, cls in tips:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:8px;
                        margin-bottom:0.6rem;font-size:0.84rem;color:#a0aec0;">
                <span>{emoji}</span><span>{msg}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Full Feature Table
    with st.expander("🔬 View Full Engineered Feature Vector", expanded=False):
        display_fv = fv.T.rename(columns={0: "Value"})
        display_fv["Value"] = display_fv["Value"].round(4)

        def color_value(val):
            try:
                v = float(val)
                if v > 0.7:   return "color: #48bb78"
                elif v < 0.2: return "color: #fc8181"
                else:          return "color: #f6ad55"
            except:
                return ""

        st.dataframe(
            display_fv.style.map(color_value) if hasattr(display_fv.style, "map") else display_fv.style.applymap(color_value),
            use_container_width=True,
            height=400
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:

    recs    = generate_recommendations(fv, inputs, prob)
    df_all  = load_student_data()
    row_fv  = fv.iloc[0]

    # ── Summary bar at top
    n_critical = sum(1 for r in recs if r["priority"] == "critical")
    n_high     = sum(1 for r in recs if r["priority"] == "high")
    n_medium   = sum(1 for r in recs if r["priority"] == "medium")
    n_low      = sum(1 for r in recs if r["priority"] == "low")

    st.markdown(f"""
    <div class="card" style="margin-bottom:1.5rem;">
        <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                    color:#e2e8f0;margin-bottom:0.8rem;">
            🎯 Recommendation Summary &nbsp;
            <span style="font-size:0.75rem;color:#4a5568;font-weight:400;">
                {len(recs)} intervention(s) identified for this student profile
            </span>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <span class="rec-priority priority-critical">🚨 {n_critical} Critical</span>
            <span class="rec-priority priority-high">⚠️ {n_high} High</span>
            <span class="rec-priority priority-medium">ℹ️ {n_medium} Medium</span>
            <span class="rec-priority priority-low">✅ {n_low} Low</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Category filter
    cat_labels = {
        "all":        "All",
        "academic":   "🎓 Academic",
        "engagement": "💻 Engagement",
        "support":    "🤝 Support",
        "pastoral":   "❤️ Pastoral",
        "admin":      "📋 Admin",
    }
    cat_css = {
        "academic":   "cat-academic",
        "engagement": "cat-engagement",
        "support":    "cat-support",
        "pastoral":   "cat-pastoral",
        "admin":      "cat-admin",
    }

    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        chosen_cat = st.selectbox(
            "Filter by Category",
            options=list(cat_labels.keys()),
            format_func=lambda k: cat_labels[k],
            label_visibility="collapsed"
        )
    with col_sort:
        show_only_critical = st.checkbox("Critical & High only", value=False)

    filtered = [r for r in recs
                if (chosen_cat == "all" or r["category"] == chosen_cat)
                and (not show_only_critical or r["priority"] in ("critical", "high"))]

    if not filtered:
        st.markdown("""
        <div class="card" style="text-align:center;color:#4a5568;padding:2rem;">
            No recommendations match the current filter.
        </div>
        """, unsafe_allow_html=True)
    else:
        col_cards, col_sidebar = st.columns([1.6, 1])

        with col_cards:
            st.markdown(f'<div class="section-title">Interventions ({len(filtered)})</div>',
                        unsafe_allow_html=True)
            for rec in filtered:
                cat_badge = f'<span class="rec-category-badge {cat_css.get(rec["category"],"")}">{cat_labels.get(rec["category"], rec["category"])}</span>'
                pri_badge = f'<span class="rec-priority priority-{rec["priority"]}">{rec["priority"].upper()}</span>'
                st.markdown(f"""
                <div class="rec-card">
                    {cat_badge}
                    <div class="rec-card-header">
                        <span class="rec-icon">{rec["icon"]}</span>
                        <span class="rec-title">{rec["title"]}</span>
                        {pri_badge}
                    </div>
                    <div class="rec-body">{rec["body"]}</div>
                    <div class="rec-action">→ {rec["action"]}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_sidebar:
            # ── Peer Comparison
            st.markdown('<div class="section-title">Peer Comparison</div>',
                        unsafe_allow_html=True)
            if not df_all.empty:
                peer = peer_comparison(fv, df_all, inputs["code_module"])
                label_map = {"Score": inputs["score"],
                             "VLE Clicks": inputs["sum_click"],
                             "Credits": inputs["studied_credits"]}
                st.markdown('<div class="peer-card">', unsafe_allow_html=True)
                for metric, stats in peer.items():
                    pct   = stats["percentile"]
                    s_val = label_map.get(metric, stats["student"])
                    avg   = stats["module_avg"]
                    color = "#48bb78" if pct >= 60 else ("#f6ad55" if pct >= 35 else "#fc8181")
                    st.markdown(f"""
                    <div class="peer-stat">
                        <span>{metric}</span>
                        <span class="peer-val" style="color:{color};">
                            {s_val:.0f}
                            <span style="font-size:0.7rem;color:#4a5568;">
                                &nbsp;(top {100-pct:.0f}% · avg {avg:.0f})
                            </span>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="peer-card" style="color:#4a5568;font-size:0.84rem;text-align:center;padding:1rem;">
                    Place <code>cleaned_student_data.csv</code> in the app directory to enable peer benchmarking.
                </div>
                """, unsafe_allow_html=True)

            # ── Action Milestones
            st.markdown('<div class="section-title" style="margin-top:1.2rem;">Action Milestones</div>',
                        unsafe_allow_html=True)

            def milestone_status(cond_good, cond_warn):
                if cond_good:   return "m-done", "✓"
                elif cond_warn: return "m-warn", "!"
                else:           return "m-todo", "✗"

            milestones = [
                ("Assessment score ≥ 55",
                 milestone_status(inputs["score"] >= 55, inputs["score"] >= 40)),
                ("Submitted on time",
                 milestone_status(row_fv["submission_timeliness"] >= 0,
                                  row_fv["submission_timeliness"] >= -2)),
                ("VLE clicks > 1 000",
                 milestone_status(inputs["sum_click"] > 1000, inputs["sum_click"] > 400)),
                ("No prior failed attempts",
                 milestone_status(inputs["num_of_prev_attempts"] == 0,
                                  inputs["num_of_prev_attempts"] == 1)),
                ("Registered early",
                 milestone_status(inputs["date_registration"] < 0,
                                  inputs["date_registration"] <= 14)),
                ("Still enrolled",
                 milestone_status(not inputs["unregistered"], False)),
                ("Credit load ≤ 120",
                 milestone_status(inputs["studied_credits"] <= 120,
                                  inputs["studied_credits"] <= 150)),
            ]

            st.markdown('<div class="card" style="padding:0.8rem 1.2rem;">', unsafe_allow_html=True)
            for label, (css_cls, symbol) in milestones:
                label_color = "#68d391" if css_cls == "m-done" else ("#f6ad55" if css_cls == "m-warn" else "#fc8181")
                st.markdown(f"""
                <div class="milestone-row">
                    <div class="milestone-check {css_cls}">{symbol}</div>
                    <span style="color:{label_color};font-size:0.83rem;">{label}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Risk level summary
            risk_level = "Critical" if prob < 0.35 else ("High" if prob < 0.50 else ("Moderate" if prob < 0.70 else "Low"))
            risk_color = {"Critical":"#fc8181","High":"#f6ad55","Moderate":"#63b3ed","Low":"#68d391"}[risk_level]
            st.markdown(f"""
            <div class="card" style="margin-top:1rem;text-align:center;padding:1rem;">
                <div style="font-size:0.72rem;color:#4a5568;letter-spacing:1px;
                            text-transform:uppercase;margin-bottom:0.4rem;">
                    Overall Risk Level
                </div>
                <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                            color:{risk_color};">{risk_level}</div>
                <div style="font-size:0.78rem;color:#4a5568;margin-top:0.3rem;">
                    Pass probability: {risk_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer
st.markdown("""
<hr>
<div style="text-align:center;color:#2d3748;font-size:0.78rem;padding:0.5rem 0;">
    EduPredict · Student Academic Performance AI · OULAD Framework ·
    <span style="color:#3d4f6e;">Built with Streamlit + Plotly</span>
</div>
""", unsafe_allow_html=True)
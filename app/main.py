import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env_file(path):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

st.set_page_config(
    page_title="ResumeIQ — Explainable Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open(os.path.join(os.path.dirname(__file__), "../assets/style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_text" not in st.session_state:
    st.session_state.job_text = ""

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧠 ResumeIQ")
    st.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "🔍 Analyze Resume": "analyzer",
        "📊 Explain Results": "explain",
        "🚀 Improve Skills": "improve",
    }
    
    for label, page_key in pages.items():
        if st.button(label, use_container_width=True, 
                     type="primary" if st.session_state.page == page_key else "secondary"):
            st.session_state.page = page_key
            st.rerun()
    
    st.markdown("---")
    st.caption("Powered by Transformers + SHAP + LLM")

# Route pages
page = st.session_state.page

if page == "home":
    from app_pages.home import render_home
    render_home()
elif page == "analyzer":
    from app_pages.analyzer import render_analyzer
    render_analyzer()
elif page == "explain":
    from app_pages.explain import render_explain
    render_explain()
elif page == "improve":
    from app_pages.improve import render_improve
    render_improve()

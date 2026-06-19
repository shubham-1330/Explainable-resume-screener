import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.matcher import ResumeMatcher
from core.explainer import ShapExplainer
from core.llm_explainer import LLMExplainer

# Sample data
SAMPLE_RESUME = """Shubham | shubhe@email.com | LinkedIn: linkedin.com/in/shubh

SUMMARY
Data scientist with 4 years of experience in machine learning, NLP, and data pipelines.
Passionate about building explainable AI systems.

EXPERIENCE
Senior Data Scientist — TechCorp (2021–Present)
- Built NLP pipelines using BERT and sentence-transformers for document classification
- Deployed ML models using FastAPI and Docker; reduced inference time by 40%
- Conducted A/B testing and statistical analysis for product decisions

Data Analyst — DataStartup (2019–2021)
- Developed dashboards in Tableau and Power BI for executive reporting
- Wrote complex SQL queries on PostgreSQL for business intelligence
- Collaborated with product teams to define KPIs and success metrics

EDUCATION
B.Tech in Computer Science — IIT Delhi (2019)

SKILLS
Python, scikit-learn, PyTorch, TensorFlow, BERT, HuggingFace Transformers,
SQL, PostgreSQL, Tableau, Docker, FastAPI, Git, AWS, Spark, Pandas, NumPy

PROJECTS
- Resume Screening System: Semantic matching with SHAP explanations
- Sentiment Analysis API: Fine-tuned DistilBERT on product reviews
"""

SAMPLE_JOB = """Senior Machine Learning Engineer — AI Innovations Inc.

ABOUT THE ROLE
We are looking for an ML Engineer with strong NLP expertise to join our AI team.
You will build, deploy, and explain ML models powering our hiring platform.

RESPONSIBILITIES
- Design and implement NLP models using transformer architectures (BERT, GPT, etc.)
- Deploy models to production using cloud infrastructure (AWS, GCP)
- Write clean, well-tested Python code with a focus on scalability
- Collaborate with product and data teams to define ML requirements
- Ensure fairness and explainability in ML systems

REQUIRED SKILLS
- 3+ years of Python and machine learning experience
- Strong knowledge of NLP and transformer models (BERT, sentence-transformers)
- Experience with MLOps: Docker, CI/CD, model serving (FastAPI, Flask)
- Proficiency with SQL and data manipulation (Pandas, NumPy)
- Familiarity with cloud platforms (AWS preferred)

NICE TO HAVE
- Experience with SHAP or other explainability frameworks
- Knowledge of fairness-aware ML
- Contributions to open-source ML projects

EDUCATION
B.Tech/B.E. in CS, EE, or related field (or equivalent experience)
"""


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        st.error("pypdf is not installed. Run: pip install pypdf")
        return ""
    except Exception as e:
        st.error(f"Failed to read PDF: {str(e)}")
        return ""


def render_analyzer():
    st.markdown("## 🔍 Resume Analyzer")
    st.markdown("Upload your resume as a **PDF** or paste it as text, then add the job description and click **Analyze**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Resume")

        # Toggle between PDF upload and text paste
        input_mode = st.radio(
            "Input method",
            ["📎 Upload PDF", "📝 Paste Text"],
            horizontal=True,
            key="resume_input_mode"
        )

        if input_mode == "📎 Upload PDF":
            uploaded_pdf = st.file_uploader(
                "Upload your resume PDF",
                type=["pdf"],
                key="resume_pdf"
            )
            if uploaded_pdf is not None:
                with st.spinner("Extracting text from PDF..."):
                    extracted = extract_text_from_pdf(uploaded_pdf)
                if extracted:
                    st.session_state.resume_text = extracted
                    st.success(f"✅ PDF loaded — {len(extracted.split())} words extracted")
                    with st.expander("Preview extracted text"):
                        st.text(extracted[:1000] + ("..." if len(extracted) > 1000 else ""))
                else:
                    st.warning("⚠️ Could not extract text. Try the Paste Text option instead.")
        else:
            if st.button("Load Sample Resume", key="load_resume"):
                st.session_state.resume_text = SAMPLE_RESUME
            resume_text = st.text_area(
                "Paste resume here...",
                value=st.session_state.get("resume_text", ""),
                height=400,
                key="resume_input",
                placeholder="Paste your full resume text here...",
            )
            st.session_state.resume_text = resume_text

    with col2:
        st.markdown("### 💼 Job Description")
        if st.button("Load Sample Job", key="load_job"):
            st.session_state.job_text = SAMPLE_JOB
        job_text = st.text_area(
            "Paste job description here...",
            value=st.session_state.get("job_text", ""),
            height=400,
            key="job_input",
            placeholder="Paste the full job description here...",
        )
        st.session_state.job_text = job_text

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button("🧠 Analyze Match", type="primary", use_container_width=True)

    if analyze_clicked:
        resume = st.session_state.resume_text.strip()
        job = st.session_state.job_text.strip()

        if not resume or not job:
            st.error("⚠️ Please provide both a resume and a job description.")
            return

        with st.spinner("🔄 Running semantic analysis..."):
            try:
                matcher = ResumeMatcher()
                result = matcher.analyze(resume, job)

                with st.spinner("🔍 Computing SHAP explanations..."):
                    explainer = ShapExplainer()
                    shap_result = explainer.explain(result)
                    result["shap"] = shap_result

                with st.spinner("💬 Generating LLM insights..."):
                    llm = LLMExplainer()
                    llm_result = llm.explain(result, resume, job)
                    result["llm_explanation"] = llm_result

                st.session_state.analysis_result = result
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
                return

        # Quick summary
        score = result.get("overall_score", 0)
        st.markdown("### 📊 Quick Results")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            st.metric(f"{color} Overall Match Score", f"{score:.1f}%")
        with col_s2:
            st.metric("✅ Matched Skills", result.get("matched_skills_count", 0))
        with col_s3:
            st.metric("❌ Missing Skills", result.get("missing_skills_count", 0))

        st.info("👉 Go to **📊 Explain Results** for detailed SHAP analysis and LLM insights.")
        if st.button("View Full Explanation →", type="primary"):
            st.session_state.page = "explain"
            st.rerun()

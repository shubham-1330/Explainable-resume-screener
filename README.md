# 🧠 Explainable Resume Screening System

A production-ready AI system that semantically matches resumes to job descriptions and **explains every decision** using SHAP, counterfactual analysis, and LLM-generated insights.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎯 **Semantic Matching** | Transformer embeddings (sentence-transformers) score alignment beyond keywords |
| 🔍 **SHAP Explanations** | Feature-level SHAP values show exactly what helped or hurt your score |
| 🔄 **Counterfactual Analysis** | "What-if" scenarios — see the minimal changes to boost your match |
| 💬 **LLM Insights** | Claude API generates plain-English, grounded career coaching |
| 📊 **Interactive Dashboard** | Streamlit UI with Plotly charts for visual exploration |

---

## 📁 Project Structure

```
explainable-resume-screener/
├── app/
│   ├── main.py                  # Streamlit entry point
│   └── pages/
│       ├── home.py              # Landing page
│       ├── analyzer.py          # Resume + JD input & analysis
│       ├── explain.py           # SHAP charts & LLM explanation
│       └── improve.py           # Counterfactuals & roadmap
├── core/
│   ├── matcher.py               # Semantic matching (sentence-transformers)
│   ├── explainer.py             # SHAP feature importance
│   └── llm_explainer.py         # Claude API integration
├── data/
│   ├── sample_resumes/          # Sample resume text files
│   └── sample_jobs/             # Sample job description files
├── utils/
│   └── text_processing.py       # Text cleaning utilities
├── assets/
│   └── style.css                # Custom Streamlit CSS
├── tests/
│   └── test_matcher.py          # Pytest unit tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / Open in VS Code

```bash
# If downloaded as zip, extract and open the folder in VS Code
code explainable-resume-screener
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90MB) on first run. This is automatic and requires internet.

### 4. (Optional) Set Anthropic API key for LLM explanations

```bash
# macOS/Linux
export ANTHROPIC_API_KEY=your_key_here

# Windows CMD
set ANTHROPIC_API_KEY=your_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_key_here"
```

> Without the API key, the system still works fully — it uses a rule-based fallback explanation. Get a key at [console.anthropic.com](https://console.anthropic.com).

### 5. Run the app

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 How It Works

### 1. Semantic Matching (`core/matcher.py`)
- Loads `all-MiniLM-L6-v2` from HuggingFace sentence-transformers
- Encodes resume and job description into 384-dim vectors
- Computes cosine similarity for overall and section-level scores
- Extracts skill tokens and scores each against the job requirements

### 2. SHAP Explanations (`core/explainer.py`)
- Builds a surrogate Ridge regression model on skill feature vectors
- Applies `shap.LinearExplainer` to compute feature-level SHAP values
- Falls back to approximate SHAP values if the library is not installed
- Returns ranked feature importance: positive = helped score, negative = hurt score

### 3. Counterfactual Analysis (`core/matcher.py`)
- Identifies missing skills and their individual impact on the match score
- Generates "what-if" scenarios: "if you add X, your score goes from Y → Z"
- Prioritizes scenarios by maximum score gain with minimum effort

### 4. LLM Explanation (`core/llm_explainer.py`)
- Sends structured analysis data to the Claude API
- Prompts Claude to generate a friendly, grounded 3-paragraph explanation
- Falls back to a template-based explanation without the API key

---

## 📊 Architecture

```
User Input (Resume + JD)
        │
        ▼
  ResumeMatcher
  ├── SentenceTransformer (all-MiniLM-L6-v2)
  ├── Cosine Similarity → Overall Score
  ├── Section Splitting → Section Scores
  └── Skill Extraction → Skill Scores
        │
        ▼
  ShapExplainer
  ├── Surrogate Ridge Model
  ├── SHAP LinearExplainer
  └── Feature Importance Dict
        │
        ▼
  LLMExplainer (Claude API)
  ├── Structured prompt with scores + SHAP
  └── Plain-English career coaching
        │
        ▼
  Streamlit UI
  ├── Gauge chart (overall score)
  ├── Bar charts (section scores, SHAP)
  ├── Skill breakdown table
  ├── Counterfactual scenarios
  └── LLM explanation + improvement plan
```

---

## 💡 Extending the System

- **Add more embedding models:** Change `model_name` in `ResumeMatcher.__init__`
- **Add PDF parsing:** Use `pypdf` or `pdfplumber` to extract text from PDF resumes
- **Add batch screening:** Extend `matcher.py` to process multiple resumes against one JD
- **Add fairness analysis:** Integrate `aif360` or `fairlearn` to audit for bias

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `sentence-transformers` | Semantic embeddings |
| `shap` | SHAP feature importance |
| `scikit-learn` | ML utilities, cosine similarity |
| `plotly` | Interactive charts |
| `anthropic` | Claude API client |
| `torch` | Required by sentence-transformers |

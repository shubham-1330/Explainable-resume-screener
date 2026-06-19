"""
LLMExplainer: Uses the Google Gemini API to generate grounded,
user-friendly explanations of resume–job match results.
"""

import os
import json
from typing import Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMExplainer:
    def __init__(self):
        self.client = None
        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")

    def explain(self, result: Dict[str, Any], resume_text: str, job_text: str) -> str:
        """Generate a plain-English explanation of the match result."""
        if self.client:
            return self._call_gemini(result, resume_text, job_text)
        else:
            return self._fallback_explanation(result)

    def _call_gemini(self, result: Dict, resume_text: str, job_text: str) -> str:
        """Call Gemini API for LLM-generated explanation."""
        score = result.get("overall_score", 0)
        matched = result.get("matched_skills", [])
        missing = [s["skill"] for s in result.get("missing_skills", [])[:5]]
        section_scores = result.get("section_scores", {})
        shap_data = result.get("shap", {})
        top_positive = sorted(
            [(k, v) for k, v in shap_data.get("feature_importance", {}).items() if v > 0],
            key=lambda x: x[1], reverse=True
        )[:3]
        top_negative = sorted(
            [(k, v) for k, v in shap_data.get("feature_importance", {}).items() if v < 0],
            key=lambda x: x[1]
        )[:3]

        prompt = f"""
You are an expert career coach analyzing a resume–job match.
Here is the analysis summary:

MATCH SCORE: {score:.1f}/100
MATCHED SKILLS: {', '.join(matched[:8]) if matched else 'None detected'}
MISSING SKILLS: {', '.join(missing) if missing else 'None'}
SECTION SCORES: {json.dumps(section_scores, indent=2)}

SHAP ANALYSIS:
- Top positive contributors (helped score): {[k for k, v in top_positive]}
- Top negative contributors (hurt score): {[k for k, v in top_negative]}

Write a friendly, specific, 3-paragraph explanation:
1. Overall assessment of the match (mention the score and what it means)
2. What's working well (specific strengths from the resume that match the job)
3. Key gaps and what the candidate should focus on

Be encouraging but honest. Use plain English, no jargon. Be specific and grounded in the data above.
Keep it under 250 words.
"""

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_explanation(result) + f"\n\n*(LLM error: {str(e)})*"

    def generate_improvement_plan(self, result: Dict) -> str:
        """Generate a personalized improvement plan."""
        if self.client:
            return self._call_improvement_plan(result)
        return self._fallback_improvement_plan(result)

    def _call_improvement_plan(self, result: Dict) -> str:
        missing_skills = result.get("missing_skills", [])
        counterfactuals = result.get("counterfactuals", [])
        score = result.get("overall_score", 0)

        prompt = f"""
You are a career coach. Based on this analysis:
- Current match score: {score:.1f}%
- Missing skills (by impact): {json.dumps([{"skill": s["skill"], "time": s["time_to_learn"]} for s in missing_skills[:5]], indent=2)}
- Counterfactual scenarios: {json.dumps([{"label": c["label"], "gain": c["projected_score"] - score} for c in counterfactuals[:3]], indent=2)}

Write a practical, motivating 3-step improvement plan (2–3 sentences per step).
Focus on the highest-impact actions. Be specific and actionable.
Keep it under 200 words.
"""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_improvement_plan(result)

    def _fallback_explanation(self, result: Dict) -> str:
        score = result.get("overall_score", 0)
        matched = result.get("matched_skills", [])
        missing = [s["skill"] for s in result.get("missing_skills", [])[:3]]

        level = "strong" if score >= 70 else "moderate" if score >= 50 else "low"
        assessment = f"Your resume shows a **{level} match** of **{score:.1f}%** with this job description. "
        if score >= 70:
            assessment += "You are a competitive candidate and should apply with confidence."
        elif score >= 50:
            assessment += "You meet many of the requirements but there are some gaps to address."
        else:
            assessment += "There are significant skill gaps to close before you'd be a strong candidate."

        strengths = ""
        if matched:
            strengths = f"\n\n**What's working:** Your resume already demonstrates strong alignment in {', '.join(matched[:5])}. These skills are clearly valued in this role."

        gaps = ""
        if missing:
            gaps = f"\n\n**Key gaps to address:** The job requires {', '.join(missing)} — skills not prominently featured in your resume. Adding these (or projects that demonstrate them) could significantly boost your match score."

        note = "\n\n*Set `GEMINI_API_KEY` in your environment to enable AI-powered personalized explanations.*"
        return assessment + strengths + gaps + note

    def _fallback_improvement_plan(self, result: Dict) -> str:
        missing = result.get("missing_skills", [])
        if not missing:
            return "Your resume is already well-aligned. Focus on tailoring the language to mirror the job description."

        steps = []
        if missing:
            top = missing[0]
            steps.append(f"**Step 1 — Priority Skill:** Learn **{top['skill']}** ({top['time_to_learn']}). This has the highest impact on your match score.")
        if len(missing) >= 2:
            steps.append(f"**Step 2 — Build a project:** Create a project that combines {missing[0]['skill']} and {missing[1]['skill']}. Add it to your GitHub and resume.")
        steps.append("**Step 3 — Tailor your resume:** Rewrite your experience bullets to mirror the job description's language. Semantic alignment matters as much as having the skills.")

        return "\n\n".join(steps) + "\n\n*Set `GEMINI_API_KEY` to get AI-personalized plans.*"
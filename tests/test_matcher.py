"""
Tests for the ResumeMatcher core module.
Run with: pytest tests/
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matcher import ResumeMatcher
from core.explainer import ShapExplainer


SAMPLE_RESUME = """
John Doe — Data Scientist
Python, machine learning, scikit-learn, SQL, Docker, AWS
Built NLP pipelines using BERT. Deployed models with FastAPI.
4 years of experience in data science.
"""

SAMPLE_JOB = """
ML Engineer Role
Requirements: Python, machine learning, NLP, BERT, Docker, FastAPI, AWS
Nice to have: SHAP, explainability, Kubernetes
"""


class TestResumeMatcher:
    def setup_method(self):
        self.matcher = ResumeMatcher()

    def test_analyze_returns_dict(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        assert isinstance(result, dict)

    def test_overall_score_in_range(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        score = result.get("overall_score", -1)
        assert 0 <= score <= 100, f"Score {score} out of range"

    def test_skill_extraction(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert isinstance(result["matched_skills"], list)

    def test_section_scores(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        section_scores = result.get("section_scores", {})
        assert isinstance(section_scores, dict)
        for v in section_scores.values():
            assert 0 <= v <= 100

    def test_counterfactuals_generated(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        cfs = result.get("counterfactuals", [])
        assert len(cfs) > 0
        for cf in cfs:
            assert "label" in cf
            assert "projected_score" in cf
            assert cf["projected_score"] >= result["overall_score"]

    def test_high_similarity_resume(self):
        """A resume that exactly mirrors the job should score higher than a random one."""
        mirror_resume = SAMPLE_JOB  # exact copy of job = perfect match
        result_mirror = self.matcher.analyze(mirror_resume, SAMPLE_JOB)
        result_random = self.matcher.analyze("I am a chef. I cook food.", SAMPLE_JOB)
        assert result_mirror["overall_score"] > result_random["overall_score"]

    def test_empty_inputs(self):
        """Should handle short/empty inputs gracefully."""
        result = self.matcher.analyze("John Doe", "Software Engineer")
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 100


class TestShapExplainer:
    def setup_method(self):
        self.matcher = ResumeMatcher()
        self.explainer = ShapExplainer()

    def test_shap_returns_dict(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        shap_result = self.explainer.explain(result)
        assert isinstance(shap_result, dict)
        assert "feature_importance" in shap_result

    def test_feature_importance_not_empty(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        shap_result = self.explainer.explain(result)
        fi = shap_result.get("feature_importance", {})
        assert len(fi) > 0

    def test_top_positive_features(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        shap_result = self.explainer.explain(result)
        top_pos = self.explainer.get_top_positive_features(shap_result)
        assert isinstance(top_pos, list)

    def test_top_negative_features(self):
        result = self.matcher.analyze(SAMPLE_RESUME, SAMPLE_JOB)
        shap_result = self.explainer.explain(result)
        top_neg = self.explainer.get_top_negative_features(shap_result)
        assert isinstance(top_neg, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

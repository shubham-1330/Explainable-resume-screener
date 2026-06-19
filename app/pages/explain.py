import streamlit as st


def render_explain():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        st.error("Plotly and pandas are required for this page. Install them with `pip install plotly pandas`.")
        return

    st.markdown("## Explainability Dashboard")

    result = st.session_state.get("analysis_result")
    if not result:
        st.warning("No analysis found. Please run the analyzer first.")
        if st.button("Go to Analyzer"):
            st.session_state.page = "analyzer"
            st.rerun()
        return

    score = result.get("overall_score", 0)
    col1, col2 = st.columns([1, 2])

    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "Match Score", "font": {"size": 20}},
            delta={"reference": 70, "increasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#4f8ef7"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("### Section-Level Scores")
        section_scores = result.get("section_scores", {})
        if section_scores:
            df_sections = pd.DataFrame(list(section_scores.items()), columns=["Section", "Score"]).sort_values(
                "Score", ascending=True
            )
            fig_bar = px.bar(
                df_sections,
                x="Score",
                y="Section",
                orientation="h",
                color="Score",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                range_color=[0, 100],
                labels={"Score": "Match Score (%)"},
            )
            fig_bar.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### SHAP Feature Importance")
    st.caption("These bars show which skills/features contributed most to your match score.")

    shap_data = result.get("shap", {})
    shap_values = shap_data.get("feature_importance", {})

    if shap_values:
        df_shap = pd.DataFrame(list(shap_values.items()), columns=["Feature", "SHAP Value"]).sort_values(
            "SHAP Value", ascending=True
        ).tail(20)

        df_shap["Color"] = df_shap["SHAP Value"].apply(lambda x: "#22c55e" if x > 0 else "#ef4444")

        fig_shap = go.Figure(go.Bar(
            x=df_shap["SHAP Value"],
            y=df_shap["Feature"],
            orientation="h",
            marker_color=df_shap["Color"],
        ))
        fig_shap.update_layout(
            height=500,
            xaxis_title="SHAP Value (impact on match score)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
        )
        fig_shap.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP values not available for this analysis.")

    st.markdown("---")
    st.markdown("### Skill-by-Skill Breakdown")
    skill_details = result.get("skill_details", [])
    if skill_details:
        df_skills = pd.DataFrame(skill_details)
        df_skills["Match"] = df_skills["score"].apply(
            lambda x: "Strong" if x >= 0.75 else ("Partial" if x >= 0.5 else "Missing")
        )
        df_skills["Score %"] = (df_skills["score"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            df_skills[["skill", "Score %", "Match"]].rename(
                columns={"skill": "Skill/Requirement", "Score %": "Similarity Score"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### AI-Generated Explanation")
    llm_text = result.get("llm_explanation", "")
    if llm_text:
        st.markdown(f'<div class="llm-box">{llm_text}</div>', unsafe_allow_html=True)
    else:
        st.info("LLM explanation not available.")

    st.markdown("---")
    if st.button("See Improvement Suggestions ->", type="primary", use_container_width=True):
        st.session_state.page = "improve"
        st.rerun()

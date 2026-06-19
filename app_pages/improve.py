import streamlit as st


def render_improve():
    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        st.error("Plotly and pandas are required for this page. Install them with `pip install plotly pandas`.")
        return

    st.markdown("## Skill Improvement Roadmap")
    st.markdown("Based on counterfactual analysis - what changes would most improve your match score?")

    result = st.session_state.get("analysis_result")
    if not result:
        st.warning("No analysis found. Please run the analyzer first.")
        if st.button("Go to Analyzer"):
            st.session_state.page = "analyzer"
            st.rerun()
        return

    score = result.get("overall_score", 0)
    counterfactuals = result.get("counterfactuals", [])
    missing_skills = result.get("missing_skills", [])

    st.markdown("### Projected Score Improvement")

    if counterfactuals:
        cf_labels = ["Current"] + [cf["label"] for cf in counterfactuals[:5]]
        cf_scores = [score] + [cf["projected_score"] for cf in counterfactuals[:5]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cf_labels,
            y=cf_scores,
            marker_color=["#94a3b8"] + ["#4f8ef7"] * len(counterfactuals[:5]),
            text=[f"{s:.1f}%" for s in cf_scores],
            textposition="outside",
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Hiring Threshold (70%)", annotation_position="right")
        fig.update_layout(
            height=350,
            yaxis=dict(range=[0, 105], title="Match Score (%)"),
            xaxis_title="Improvement Scenario",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Priority Skill Gaps")
    st.caption("Skills from the job description that are missing or weak in your resume, ranked by impact.")

    if missing_skills:
        for skill_info in missing_skills[:10]:
            impact = skill_info.get("impact", 0)
            skill = skill_info.get("skill", "")
            level = skill_info.get("level_required", "intermediate")
            resources = skill_info.get("resources", [])

            with st.expander(f"{skill} - Impact: +{impact:.1f}% if added"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Required Level:** {level.title()}")
                    st.markdown(f"**Estimated Time to Learn:** {skill_info.get('time_to_learn', '2-4 weeks')}")
                with col2:
                    if resources:
                        st.markdown("**Learning Resources:**")
                        for r in resources:
                            st.markdown(f"- [{r['name']}]({r['url']})")
    else:
        st.success("Great news - no major skill gaps detected!")

    st.markdown("---")
    st.markdown("### Counterfactual Scenarios")
    st.caption("Concrete what-if scenarios that would meaningfully improve your match score.")

    if counterfactuals:
        for cf in counterfactuals:
            delta = cf["projected_score"] - score
            st.markdown(
                f"""
                <div class="cf-card">
                    <div class="cf-header">
                        <strong>{cf['label']}</strong>
                        <span class="cf-badge">+{delta:.1f}% -> {cf['projected_score']:.1f}%</span>
                    </div>
                    <p>{cf['description']}</p>
                    <ul>{"".join(f"<li>{a}</li>" for a in cf.get('actions', []))}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Personalized Improvement Plan")
    improvement_plan = result.get("improvement_plan", "")
    if improvement_plan:
        st.markdown(f'<div class="llm-box">{improvement_plan}</div>', unsafe_allow_html=True)
    else:
        st.info("Run the full analysis to generate a personalized improvement plan.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Explanation", use_container_width=True):
            st.session_state.page = "explain"
            st.rerun()
    with col2:
        if st.button("Analyze Another Resume", type="primary", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.resume_text = ""
            st.session_state.job_text = ""
            st.session_state.page = "analyzer"
            st.rerun()

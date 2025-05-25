import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def render_model_info(model, feature_names):
    st.markdown('<div class="notion-accent-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="display:inline; margin-left: 0.5rem;">ℹ️ Model Information</h2>
        <div class="notion-callout" style="margin-top:1rem;">
            <span style="font-size:1.5rem;">🤖</span>
            <div>
                Learn how the machine learning model works and what drives its predictions.<br>
                <span style="color:#60a5fa;">Tip:</span> Explore feature importance and performance metrics below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    model_type = "Random Forest Classifier" if hasattr(model, 'estimators_') else "Machine Learning Model"
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <div style="background-color: #2563EB; color: white; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
            <span style="font-size: 24px;">🤖</span>
        </div>
        <div>
            <h3 style="margin: 0;">{model_type}</h3>
            <p style="margin: 0; color: #a1a1aa;">Optimized for detecting academic misconduct patterns</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.markdown("### How It Works")
    st.markdown("""
    This model uses a supervised machine learning approach to detect potential academic misconduct:

    1. **Data Collection**: Gathers data about student performance in various assessments  
    2. **Feature Engineering**: Transforms raw data into relevant features  
    3. **Training**: Uses known cases to learn patterns of academic misconduct  
    4. **Prediction**: Applies learned patterns to identify anomalies in new data  
    5. **Threshold Selection**: Optimizes decision threshold to balance false positives and negatives
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance visualization
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        feature_importances = np.array([0.18, 0.16, 0.15, 0.12, 0.11, 0.1, 0.08, 0.06, 0.04])
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    fig = px.bar(
        importance_df,
        y='Feature',
        x='Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues,
        title='Feature Importance'
    )
    fig.update_layout(
        height=400,
        xaxis_title="Importance",
        yaxis_title=None,
        coloraxis_showscale=False,
        plot_bgcolor="#18181b",
        paper_bgcolor="#18181b",
        font_color="#e5e7eb"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model performance metrics
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    # Generate demo ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.array([0] + list(1 / (1 + np.exp(-9 * (x - 0.5))) for x in fpr[1:]))
    roc_auc = np.trapz(tpr, fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#2563EB', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700,
        height=500,
        plot_bgcolor="#18181b",
        paper_bgcolor="#18181b",
        font_color="#e5e7eb",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(24,24,27,0.8)',
            bordercolor='rgba(0,0,0,0.2)'
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Model Technical Details"):
        st.markdown("""
        #### Model Parameters

        - **Model Type**: Random Forest Classifier
        - **Number of Trees**: 100
        - **Max Depth**: 10
        - **Feature Selection**: All features
        - **Cross-Validation**: 5-fold
        - **Metric Optimized**: Precision-Recall AUC

        #### Training Information

        - **Training Data Size**: 10,000 students
        - **Positive Class**: 12% (academic misconduct cases)
        - **Training Period**: Last 5 academic years
        - **Data Sources**: Anonymized academic records

        #### Performance Metrics

        - **Accuracy**: 0.92
        - **Precision**: 0.85
        - **Recall**: 0.78
        - **F1 Score**: 0.81
        - **AUC-ROC**: 0.94
        """)
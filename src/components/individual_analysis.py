import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.analysis import analyze_student, create_radar_chart
from utils.feature_info import get_feature_explanation
from utils.data_generation import generate_demo_data
from utils.metrics import calculate_metrics

def render_individual_analysis(model, scaler, threshold, feature_names):
    st.header("Individual Student Analysis")
    st.markdown("Analyze individual students with detailed visualizations and feature contributions.")
    
    # Option to use demo data or input values manually
    use_demo = st.checkbox("Use demo data", value=True)
    
    if use_demo:
        from utils.data_generation import generate_demo_data
        # Load demo data
        X_demo, y_demo = generate_demo_data(100)
        
        # Select a student from the demo data
        student_options = X_demo['student_id'].tolist()
        student_id = st.selectbox("Select a student ID:", student_options)
        
        student_idx = X_demo[X_demo['student_id'] == student_id].index[0]
        student_data = X_demo.drop('student_id', axis=1).iloc[student_idx].values
        actual_label = y_demo.iloc[student_idx]
    else:
        from utils.metrics import calculate_metrics
        # Input raw data
        st.subheader("Enter Raw Student Data:")
        coursework_scores = st.text_input("Coursework Scores (comma-separated)", "85, 90, 88, 92")
        exam_scores = st.text_input("Exam Scores (comma-separated)", "80, 85, 83, 87")
        historical_scores = st.text_input("Historical Scores (comma-separated)", "75, 78, 80, 82")
        exam_times = st.text_input("Exam Times (comma-separated)", "50, 55, 53, 52")
        peer_group_scores = st.text_input("Peer Group Exam Scores (comma-separated)", "82, 84, 83, 85")
    
        # Convert input strings to lists of floats
        coursework_scores = list(map(float, coursework_scores.split(',')))
        exam_scores = list(map(float, exam_scores.split(',')))
        historical_scores = list(map(float, historical_scores.split(',')))
        exam_times = list(map(float, exam_times.split(',')))
        peer_group_scores = list(map(float, peer_group_scores.split(',')))
    
        # Calculate metrics
        metrics = calculate_metrics(coursework_scores, exam_scores, historical_scores, exam_times, peer_group_scores)
        student_data = [metrics[feature] for feature in feature_names]
    
    # Analyze button
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        analyze_button = st.button("Analyze Student", type="primary", use_container_width=True)
    
    if analyze_button:
        # Get prediction and contributions
        prediction, prob, feature_contributions = analyze_student(
            student_data, model, feature_names, threshold
        )
        
        # Display results in cards
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("Prediction Result")
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of Academic Misconduct"},
                gauge={
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2563EB" if prob <= threshold else "#EF4444"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, threshold], 'color': '#DBEAFE'},
                        {'range': [threshold, 1], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1:
                st.markdown('### ⚠️ Potential Academic Misconduct Detected')
                st.markdown('This student\'s performance pattern shows anomalies that warrant further investigation.')
            else:
                st.markdown('### ✅ No Anomaly Detected')
                st.markdown('This student\'s performance pattern appears consistent with expected behavior.')
            
            # Feature contributions
            st.subheader("Top Contributing Factors")
            contributions_df = pd.DataFrame({
                'Feature': list(feature_contributions.keys()),
                'Contribution': list(feature_contributions.values()),
                'Absolute': np.abs(list(feature_contributions.values()))
            }).sort_values('Absolute', ascending=False).head(5)
            
            fig = px.bar(
                contributions_df,
                y='Feature',
                x='Contribution',
                orientation='h',
                color='Contribution',
                color_continuous_scale=['green', 'yellow', 'red'],
                title='Top 5 Features Contributing to Prediction'
            )
            
            fig.update_layout(
                height=350,
                xaxis_title="Contribution to Anomaly Score",
                yaxis_title=None,
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Visualization")
            
            # Create interactive radar chart
            radar_fig = create_radar_chart(student_data, feature_names)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Explanation of features
            st.subheader("Key Features Explanation")
            
            # Get top 3 contributing features
            top_features = contributions_df['Feature'].head(3).tolist()
            
            for feature in top_features:
                feature_info = get_feature_explanation(feature)
                st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #1F2937;">
                    <h4>{feature_info['icon']} {feature_info['title']}</h4>
                    <p>{feature_info['desc']}</p>
                    <p><strong>Value:</strong> {student_data[feature_names.index(feature)]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show button to see all features
            with st.expander("See all feature explanations"):
                for feature in feature_names:
                    feature_info = get_feature_explanation(feature)
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #1F2937;">
                        <h4>{feature_info['icon']} {feature_info['title']}</h4>
                        <p>{feature_info['desc']}</p>
                        <p><strong>Value:</strong> {student_data[feature_names.index(feature)]:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
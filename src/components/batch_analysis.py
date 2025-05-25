import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_generation import generate_demo_data

def render_batch_analysis(model, scaler, threshold, feature_names):
    st.header("Batch Analysis")
    st.markdown("Process multiple students at once and identify potential anomalies.")
    
    st.subheader("Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with student data",
        type=["csv"],
        help="Upload a CSV file containing the required features for each student."
    )
    
    # Add data template download option
    st.markdown("##### Download Template")
    template_df = pd.DataFrame(columns=feature_names + ['student_id'])
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "Download CSV Template",
        data=template_csv,
        file_name="anomaly_detection_template.csv",
        mime="text/csv",
    )
    
    st.markdown("##### Required Columns")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px;">
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>coursework_z</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>exam_z</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>z_diff</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>score_variance</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>exam_time_std</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>peer_comparison</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>subject_variation</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>historical_trend</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>anomaly_score</code></div>
        <div style="background-color: #1F2937; color: white; padding: 8px; border-radius: 4px;"><code>student_id</code> (optional)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process uploaded file or demo data
    if uploaded_file is not None:
        process_data(uploaded_file, model, threshold, feature_names)
    else:
        st.markdown("""
        <div class="info-box" style="background-color: #1F2937">
            <h3>No data to upload?</h3>
            <p>Try our demo data to see how batch analysis works.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run with Demo Data"):
            X_demo, y_demo = generate_demo_data(100)
            process_data(X_demo, model, threshold, feature_names, is_demo=True)

def process_data(data, model, threshold, feature_names, is_demo=False):
    try:
        if not is_demo:
            df = pd.read_csv(data)
            missing_cols = [col for col in feature_names if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
        else:
            df = data
            
        # Extract features
        X = df[feature_names].values
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = np.clip(0.3 + np.mean(np.abs(X), axis=1) / 3, 0, 1)
            
        predictions = (probs > threshold).astype(int)
        
        # Add results to dataframe
        results_df = df.copy()
        results_df['Cheating Probability'] = probs
        results_df['Prediction'] = predictions
        results_df['Flag'] = results_df['Prediction'].apply(lambda x: '⚠️' if x == 1 else '✅')
        
        display_results(results_df, threshold, feature_names)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.markdown("Please check that your file has the required columns and format.")

def display_results(results_df, threshold, feature_names):
    st.subheader("Analysis Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(results_df))
    col2.metric("Flagged Students", sum(results_df['Prediction']))
    col3.metric("Flagged Percentage", f"{(sum(results_df['Prediction']) / len(results_df) * 100):.1f}%")
    col4.metric("Threshold", f"{threshold:.2f}")
    
    # Filter options
    st.subheader("Filter Results")
    min_prob = st.slider("Minimum Probability", 0.0, 1.0, threshold, 0.01)
    
    # Apply filters
    filtered_df = results_df[results_df['Cheating Probability'] >= min_prob]
    
    # Display filtered results
    st.subheader(f"Students with Probability >= {min_prob:.2f}")
    if len(filtered_df) > 0:
        display_df = filtered_df.copy()
        if 'student_id' in display_df.columns:
            display_cols = ['student_id', 'Flag', 'Cheating Probability', 'Prediction'] + feature_names
        else:
            display_df['student_id'] = [f"Student {i+1}" for i in range(len(display_df))]
            display_cols = ['student_id', 'Flag', 'Cheating Probability', 'Prediction'] + feature_names
        
        display_df = display_df[display_cols]
        display_df['Cheating Probability'] = display_df['Cheating Probability'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            display_df,
            column_config={
                "Flag": st.column_config.TextColumn(
                    "Status",
                    help="Flag indicating potential academic misconduct"
                ),
                "Cheating Probability": st.column_config.TextColumn(
                    "Probability",
                    help="Probability of academic misconduct"
                ),
                "Prediction": st.column_config.NumberColumn(
                    "Prediction (1=Flagged)",
                    help="Binary prediction (1 = potential misconduct, 0 = no anomaly)",
                    format="%d"
                )
            },
            use_container_width=True
        )
    else:
        st.info("No students match the current filter criteria.")
    
    # Visualization of probabilities
    st.subheader("Probability Distribution")
    fig = px.histogram(
        results_df,
        x='Cheating Probability',
        nbins=20,
        marginal="rug",
        opacity=0.7,
        color_discrete_sequence=["#2563EB"]
    )
    
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Distribution of Cheating Probabilities",
        xaxis_title="Probability of Academic Misconduct",
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export option
    csv = results_df.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv",
    )
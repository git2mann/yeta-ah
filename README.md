# Exam Anomaly Detection System

This web application provides an interface for detecting potential exam anomalies using machine learning. The system analyzes student performance patterns to identify potential cases of academic misconduct.

## Features

- **Individual Student Analysis**: Analyze individual students with detailed visualizations
- **Batch Analysis**: Process multiple students at once and identify potential anomalies
- **Interactive Visualizations**: View radar charts, feature contributions, and probability distributions
- **Model Information**: Understand the model's methodology and feature importance

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Access the application in your web browser at `http://localhost:8501`

## Data Requirements

For batch analysis, upload a CSV file with the following columns:
- coursework_z
- exam_z
- z_diff
- score_variance
- exam_time_std
- peer_comparison
- subject_variation
- historical_trend
- anomaly_score

## Model Information

The system uses a Random Forest classifier trained on patterns of academic performance. It identifies potential anomalies based on:

1. Inconsistent performance between coursework and exams
2. Unusual patterns across different assessments
3. Suspicious timing during exams
4. Sudden improvements compared to historical performance
5. Performance that significantly deviates from peer group

## Important Note

This system is designed as a screening tool to identify potential cases for further review, not as a definitive determination of academic misconduct. All flagged cases should be reviewed by academic staff following appropriate institutional policies.
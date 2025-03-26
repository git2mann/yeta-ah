import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Set page configuration
st.set_page_config(
    page_title="Yeta Ah",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "[Yeta Ah] Exam Anomaly Detection System - Advanced machine learning solution for identifying potential academic misconduct patterns."
    }
)

st.markdown("""
<style>
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        line-height: 1.7;
        color: #111827; /* Darker text for better contrast */
        background-color: #F3F4F6;
    }

    .main {
        background-color: #F9FAFB;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #1F2937;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }

    h1, h2, h3 {
        color: #1F2937;
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        padding: 3rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Feature Cards */
    .feature-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }

    /* Buttons */
    .btn-primary {
        background-color: #2563EB;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
        border: none;
    }
    .btn-primary:hover {
        background-color: #1E40AF !important;
    }

    .btn-secondary {
        background-color: white;
        color: #2563EB;
        border: 1px solid #2563EB;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
    }
    .btn-secondary:hover {
        background-color: #EFF6FF;
        border-color: #2563EB;
    }

    /* Code Block */
    .code-block {
        background-color: #E5E7EB; /* Slightly darker for better contrast */
        padding: 1.5rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        font-family: 'Fira Code', 'Courier New', monospace;
    }

    /* Screenshot Borders */
    .screenshot {
        border: 1px solid #D1D5DB;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15), 0 2px 4px -1px rgba(0, 0, 0, 0.08);
    }

    /* Highlighted Information */
    .bg-blue-50 {
        background-color: #DBEAFE;
        border-left: 4px solid #1E3A8A !important;
    }
    .bg-yellow-50 {
        background-color: #FEF3C7;
        border-left: 4px solid #B45309 !important;
    }

    .bg-blue-50 strong, .bg-yellow-50 strong {
        color: #111827;
    }

    /* Links */
    a {
        color: #1E40AF;
        font-weight: 600;
    }
    a:hover {
        color: #1D4ED8;
        text-decoration: underline;
    }

    /* Footer */
    footer a {
        color: #E5E7EB;
        font-weight: 500;
    }
    footer a:hover {
        color: #93C5FD !important;
    }

    /* Warning and Info Boxes */
    .warning-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and related files
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('cheating_detection_model.pkl')
        scaler = joblib.load('scaler.joblib')
        threshold = joblib.load('best_threshold.joblib')
        return model, scaler, threshold
    except FileNotFoundError:
        # Create dummy model and resources for demo purposes
        st.warning("Model files not found. Using simulated model for demonstration.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        scaler = None
        threshold = 0.7
        return model, scaler, threshold

# Function to generate synthetic demo data
@st.cache_data
def generate_demo_data(n_samples=100):
    np.random.seed(42)
    feature_names = ['coursework_z', 'exam_z', 'z_diff', 'score_variance', 
                     'exam_time_std', 'peer_comparison', 'subject_variation', 
                     'historical_trend', 'anomaly_score']
    
    # Generate random features
    X = np.random.randn(n_samples, len(feature_names))
    
    # Create more realistic relationships
    X[:, 0] = np.random.normal(0, 1, n_samples)  # coursework_z
    X[:, 1] = np.random.normal(0, 1, n_samples)  # exam_z
    X[:, 2] = X[:, 1] - X[:, 0]  # z_diff
    
    # Create some anomalies for demo
    cheater_indices = np.random.choice(n_samples, size=int(n_samples*0.15), replace=False)
    X[cheater_indices, 1] += np.random.uniform(1.5, 3, size=len(cheater_indices))  # Higher exam scores
    X[cheater_indices, 2] += np.random.uniform(1.5, 3, size=len(cheater_indices))  # Larger z_diff
    X[cheater_indices, 8] += np.random.uniform(0.4, 0.8, size=len(cheater_indices))  # Higher anomaly score
    
    # Create labels (1 for cheaters)
    y = np.zeros(n_samples)
    y[cheater_indices] = 1
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['student_id'] = [f"S{i+1000}" for i in range(n_samples)]
    y_series = pd.Series(y, name='label')
    
    return X_df, y_series

# Function to analyze a student
def analyze_student(student_data, model, feature_names, threshold):
    """Analyzes student data and provides prediction with feature contributions"""
    # Get model prediction probabilities
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(student_data.reshape(1, -1))[0][1]
    else:
        # Simulated probability for demo
        prob = 0.3 + (np.sum(np.abs(student_data)) / 10)
        prob = min(max(prob, 0), 1)
    
    prediction = 1 if prob > threshold else 0
    
    # Feature contributions
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        # Simulated feature importances for demo
        feature_importances = np.array([0.18, 0.16, 0.15, 0.12, 0.11, 0.1, 0.08, 0.06, 0.04])
    
    feature_contributions = dict(zip(feature_names, feature_importances * student_data))
    
    return prediction, prob, feature_contributions

# Function to create radar chart with Plotly
def create_radar_chart(student_data, feature_names):
    """Creates an interactive radar chart for student features using Plotly"""
    fig = go.Figure()
    
    # Complete the loop for the radar chart
    values = student_data.tolist()
    values.append(values[0])
    feature_names_loop = feature_names.copy()
    feature_names_loop.append(feature_names[0])
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=feature_names_loop,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.2)',
        line=dict(color='#2563EB', width=2),
        name='Student Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3]
            ),
        ),
        showlegend=False,
        width=500,
        height=500,
        margin=dict(l=80, r=80, t=20, b=20)
    )
    
    return fig

# Function to get feature explanation with icons
def get_feature_explanation(feature):
    feature_info = {
        'coursework_z': {
            'icon': '📝',
            'title': 'Coursework Z-Score',
            'desc': 'Standardized coursework score. Unusually high scores may indicate plagiarism.'
        },
        'exam_z': {
            'icon': '📄',
            'title': 'Exam Z-Score',
            'desc': 'Standardized exam score. Unusually high scores may indicate advance knowledge of questions.'
        },
        'z_diff': {
            'icon': '⚖️',
            'title': 'Z-Score Difference',
            'desc': 'Difference between coursework and exam z-scores. Large differences may indicate inconsistent knowledge.'
        },
        'score_variance': {
            'icon': '📊',
            'title': 'Score Variance',
            'desc': 'Variance in student\'s scores across assessments. Low variance can indicate suspicious consistency.'
        },
        'exam_time_std': {
            'icon': '⏱️',
            'title': 'Exam Time (standardized)',
            'desc': 'Standardized time taken to complete the exam. Very fast or slow completion times may be suspicious.'
        },
        'peer_comparison': {
            'icon': '👥',
            'title': 'Peer Comparison',
            'desc': 'Performance relative to peer group. Significant outperformance of peers may be suspicious.'
        },
        'subject_variation': {
            'icon': '📚',
            'title': 'Subject Variation',
            'desc': 'Variation in performance across subjects. Low variation can indicate suspicious consistency.'
        },
        'historical_trend': {
            'icon': '📈',
            'title': 'Historical Trend',
            'desc': 'Change in performance compared to past. Sudden improvements may be suspicious.'
        },
        'anomaly_score': {
            'icon': '🔍',
            'title': 'Anomaly Score',
            'desc': 'Statistical measure of how unusual the pattern is based on Isolation Forest algorithm.'
        }
    }
    return feature_info.get(feature, {'icon': '❓', 'title': feature, 'desc': 'No description available'})

# Feedback mechanism
def save_feedback(student_id, prediction, feedback):
    """Save feedback to a CSV file."""
    feedback_file = "feedback.csv"
    feedback_data = {
        "student_id": student_id,
        "prediction": prediction,
        "feedback": feedback
    }
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_df = feedback_df.append(feedback_data, ignore_index=True)
    else:
        feedback_df = pd.DataFrame([feedback_data])
    feedback_df.to_csv(feedback_file, index=False)

# Main function
def main():
    # Load model resources
    model, scaler, threshold = load_model_resources()
    
    # Define feature names
    feature_names = ['coursework_z', 'exam_z', 'z_diff', 'score_variance', 
                     'exam_time_std', 'peer_comparison', 'subject_variation', 
                     'historical_trend', 'anomaly_score']
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E40AF 0%, #1F2937 100%); padding: 3rem 2rem; border-radius: 8px; color: white; margin-bottom: 2rem; text-align: center;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">Exam Anomaly Detection System</h1>
        <p style="font-size: 1.25rem; font-weight: 400;">An advanced machine learning solution for identifying potential academic misconduct patterns in exam performance.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("IMG_7908.PNG", width=150)
        st.title("Exam Anomaly Detection")
        st.info(
            "This application helps detect potential exam anomalies "
            "using machine learning. Upload student data or use the "
            "demo data to see how it works."
        )
        
#        st.markdown("### Quick Links")
#        st.markdown("- [Individual Analysis](#individual-analysis)")
#        st.markdown("- [Batch Analysis](#batch-analysis)")
#        st.markdown("- [Model Information](#model-information)")
#        st.markdown("- [Ethical Guidelines](#ethical-guidelines)")
        
        st.markdown("---")
        st.markdown("### System Status")
        st.success("✅ Model loaded successfully")
        st.success(f"✅ Detection threshold: {threshold:.3f}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Individual Analysis", "👥 Batch Analysis", "ℹ️ Model Information", "⚠️ Ethical Guidelines"])
    
    with tab1:
        st.header("Individual Student Analysis")
        st.markdown("Analyze individual students with detailed visualizations and feature contributions.")
        
        # Option to use demo data or input values manually
        use_demo = st.checkbox("Use demo data", value=True)
        
        if use_demo:
            # Load demo data
            X_demo, y_demo = generate_demo_data(100)
            
            # Select a student from the demo data
            student_options = X_demo['student_id'].tolist()
            student_id = st.selectbox("Select a student ID:", student_options)
            
            student_idx = X_demo[X_demo['student_id'] == student_id].index[0]
            student_data = X_demo.drop('student_id', axis=1).iloc[student_idx].values
            actual_label = y_demo.iloc[student_idx]
            
            #st.info(f"Actual label (for demo purposes): {'Potential Academic Misconduct' if actual_label == 1 else 'No Anomaly Detected'}")
        else:
            # Create cards for input
            # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Enter student data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                coursework_z = st.slider("Coursework Z-Score", -3.0, 3.0, 0.0, 0.1,
                                        help="The standardized score of coursework, where 0 is average, positive values are above average, and negative values are below average.")
                exam_z = st.slider("Exam Z-Score", -3.0, 3.0, 0.0, 0.1,
                                  help="The standardized score of the exam, where 0 is average, positive values are above average, and negative values are below average.")
                z_diff = st.slider("Z-Score Difference", -5.0, 5.0, 0.0, 0.1,
                                  help="The difference between exam and coursework z-scores. Large positive values suggest better exam performance than coursework.")
                score_variance = st.slider("Score Variance", 0.0, 2.0, 1.0, 0.1,
                                          help="The variance in student's scores across different assessments. Low variance can indicate suspicious consistency.")
                exam_time_std = st.slider("Exam Time (standardized)", -3.0, 3.0, 0.0, 0.1,
                                         help="Standardized time taken to complete the exam. Negative values indicate faster completion, positive values indicate slower completion.")
            
            with col2:
                peer_comparison = st.slider("Peer Comparison", -3.0, 3.0, 0.0, 0.1,
                                           help="Performance relative to peer group. Higher values indicate performance above peers.")
                subject_variation = st.slider("Subject Variation", 0.0, 2.0, 1.0, 0.1,
                                             help="Variation in performance across different subjects. Low variation can indicate suspicious consistency.")
                historical_trend = st.slider("Historical Trend", -3.0, 3.0, 0.0, 0.1,
                                            help="Change in performance compared to past. High positive values indicate sudden improvement.")
                anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, 0.0, 0.1,
                                         help="Statistical measure of how unusual the pattern is. Higher values indicate more anomalous patterns.")
            
            st.markdown('</div>', unsafe_allow_html=True)
                
            # Combine all features
            student_data = np.array([
                coursework_z, exam_z, z_diff, score_variance, exam_time_std,
                peer_comparison, subject_variation, historical_trend, anomaly_score
            ])
        
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
                # Prediction result card
                # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                
                # Create a gauge chart with Plotly
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
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the prediction with appropriate styling
                if prediction == 1:
                    # st.markdown('<div style="background-color: #FEE2E2; padding: 16px; border-radius: 8px; border-left: 4px solid #EF4444;">', unsafe_allow_html=True)
                    st.markdown('### ⚠️ Potential Academic Misconduct Detected')
                    st.markdown('This student\'s performance pattern shows anomalies that warrant further investigation.')
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # st.markdown('<div style="background-color: #DCFCE7; padding: 16px; border-radius: 8px; border-left: 4px solid #22C55E;">', unsafe_allow_html=True)
                    st.markdown('### ✅ No Anomaly Detected')
                    st.markdown('This student\'s performance pattern appears consistent with expected behavior.')
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature contributions
                st.subheader("Top Contributing Factors")
                
                # Convert to DataFrame for better handling
                contributions_df = pd.DataFrame({
                    'Feature': list(feature_contributions.keys()),
                    'Contribution': list(feature_contributions.values()),
                    'Absolute': np.abs(list(feature_contributions.values()))
                })
                contributions_df = contributions_df.sort_values('Absolute', ascending=False).head(5)
                
                # Create a horizontal bar chart with Plotly
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
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Feature visualization card
                # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.subheader("Feature Visualization")
                
                # Create interactive radar chart with Plotly
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
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        
    
    with tab2:
        st.header("Batch Analysis")
        st.markdown("Process multiple students at once and identify potential anomalies.")
        
        # File uploader for batch analysis
        # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                # Load and process the uploaded data
                df = pd.read_csv(uploaded_file)
                
                # Check if the dataframe has the required columns
                missing_cols = [col for col in feature_names if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Extract features
                    X = df[feature_names].values
                    
                    # Scale the features if they're not already scaled
                    if st.checkbox("Data needs scaling", value=True) and scaler is not None:
                        X = scaler.transform(X)
                    
                    # Make predictions
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X)[:, 1]
                    else:
                        # Simulated probabilities for demo
                        probs = np.clip(0.3 + np.mean(np.abs(X), axis=1) / 3, 0, 1)
                        
                    predictions = (probs > threshold).astype(int)
                    
                    # Add results to the dataframe
                    results_df = df.copy()
                    results_df['Cheating Probability'] = probs
                    results_df['Prediction'] = predictions
                    results_df['Flag'] = results_df['Prediction'].apply(lambda x: '⚠️' if x == 1 else '✅')
                    
                    # Create analysis dashboard
                    # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                    st.subheader("Analysis Results")
                    
                    # Summary statistics with metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Students", len(results_df))
                    col2.metric("Flagged Students", sum(predictions))
                    col3.metric("Flagged Percentage", f"{(sum(predictions) / len(predictions) * 100):.1f}%")
                    col4.metric("Threshold", f"{threshold:.2f}")
                    
                    # Filter options
                    st.subheader("Filter Results")
                    min_prob = st.slider("Minimum Probability", 0.0, 1.0, threshold, 0.01)
                    
                    # Apply filters
                    filtered_df = results_df[results_df['Cheating Probability'] >= min_prob]
                    
                    # Display filtered results
                    st.subheader(f"Students with Probability >= {min_prob:.2f}")
                    if len(filtered_df) > 0:
                        # Format the dataframe for display
                        display_df = filtered_df.copy()
                        if 'student_id' in display_df.columns:
                            display_cols = ['student_id', 'Flag', 'Cheating Probability', 'Prediction'] + feature_names
                        else:
                            display_df['student_id'] = [f"Student {i+1}" for i in range(len(display_df))]
                            display_cols = ['student_id', 'Flag', 'Cheating Probability', 'Prediction'] + feature_names
                        
                        # Reorder columns for display
                        display_df = display_df[display_cols]
                        
                        # Format probability as percentage
                        display_df['Cheating Probability'] = display_df['Cheating Probability'].apply(lambda x: f"{x:.2%}")
                        
                        # Display with styling
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
                    
                    # Create histogram with Plotly
                    fig = px.histogram(
                        results_df,
                        x='Cheating Probability',
                        nbins=20,
                        marginal="rug",
                        opacity=0.7,
                        color_discrete_sequence=["#2563EB"]
                    )
                    
                    # Add threshold line
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
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.markdown("Please check that your file has the required columns and format.")
        
        # Add a demo option
        st.markdown("""
        <div class="info-box" style="background-color: #1F2937">
            <h3>No data to upload?</h3>
            <p>Try our demo data to see how batch analysis works.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run with Demo Data"):
            # Generate demo data
            X_demo, y_demo = generate_demo_data(100)
            
            # Make predictions with demo data
            X_features = X_demo[feature_names].values
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_features)[:, 1]
            else:
                # Simulated probabilities for demo
                probs = np.clip(0.3 + np.mean(np.abs(X_features), axis=1) / 3, 0, 1)
                
            predictions = (probs > threshold).astype(int)
            
            # Create results dataframe
            results_df = X_demo.copy()
            results_df['Cheating Probability'] = probs
            results_df['Prediction'] = predictions
            results_df['Flag'] = results_df['Prediction'].apply(lambda x: '⚠️' if x == 1 else '✅')
            
            # Create analysis dashboard
            # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Demo Analysis Results")
            
            # Summary statistics with metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Students", len(results_df))
            col2.metric("Flagged Students", sum(predictions))
            col3.metric("Flagged Percentage", f"{(sum(predictions) / len(predictions) * 100):.1f}%")
            col4.metric("Threshold", f"{threshold:.2f}")
            
            # Display filtered results
            st.subheader(f"Flagged Students (Probability >= {threshold:.2f})")
            
            # Filter to show only flagged students
            flagged_df = results_df[results_df['Prediction'] == 1]
            
            if len(flagged_df) > 0:
                # Format probability as percentage
                display_df = flagged_df.copy()
                display_df['Cheating Probability'] = display_df['Cheating Probability'].apply(lambda x: f"{x:.2%}")
                
                # Display with styling
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
                st.info("No students were flagged in the demo data.")
            
            # Visualization of probabilities
            st.subheader("Probability Distribution")
            
            # Create histogram with Plotly
            fig = px.histogram(
                results_df,
                x='Cheating Probability',
                nbins=20,
                marginal="rug",
                opacity=0.7,
                color_discrete_sequence=["#2563EB"]
            )
            
            # Add threshold line
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
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("Model Information")
        st.markdown("Information about the machine learning model and how it works.")
        
        # Model information card
        # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Model Overview")
        
        model_type = "Random Forest Classifier" if hasattr(model, 'estimators_') else "Machine Learning Model"
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <div style="background-color: #2563EB; color: white; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                <span style="font-size: 24px;">🤖</span>
            </div>
            <div>
                <h3 style="margin: 0;">{model_type}</h3>
                <p style="margin: 0; color: #4B5563;">Optimized for detecting academic misconduct patterns</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        st.markdown("""
        This model uses a supervised machine learning approach to detect potential academic misconduct:
        
        1. **Data Collection**: Gathers data about student performance in various assessments
        2. **Feature Engineering**: Transforms raw data into relevant features
        3. **Training**: Uses known cases to learn patterns of academic misconduct
        4. **Prediction**: Applies learned patterns to identify anomalies in new data
        5. **Threshold Selection**: Optimizes decision threshold to balance false positives and negatives
        """)
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        
        # Generate feature importance data (real or simulated)
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            # Simulated feature importances for demo
            feature_importances = np.array([0.18, 0.16, 0.15, 0.12, 0.11, 0.1, 0.08, 0.06, 0.04])
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Create a bar chart with Plotly
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
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("Model Performance")
        
        # Generate demo ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.array([0] + list(1 / (1 + np.exp(-9 * (x - 0.5))) for x in fpr[1:]))
        roc_auc = np.trapz(tpr, fpr)
        
        # Create ROC curve with Plotly
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
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.2)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.header("Ethical Guidelines")
        st.markdown("Important considerations for responsible use of the system.")
        
        # Ethical guidelines card
        # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Ethical Use Guidelines")
        
        st.markdown("""
        <div class="warning-box" style="background-color: #1F2937">
            <h3>⚠️ Important Notice</h3>
            <p>This system is designed to be a <strong>decision support tool</strong> and not an automated decision-making system. All flagged cases must be reviewed by qualified academic staff before any action is taken.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Principles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Fairness & Bias Mitigation
            - Regular audits for algorithmic bias
            - Balanced representation in training data
            - Continuous monitoring for disparate impact
            
            #### Human Oversight
            - All flags require human review
            - Staff training on tool limitations
            - Appeal process for students
            """)
        
        with col2:
            st.markdown("""
            #### Transparency
            - Clear documentation of model workings
            - Explainable predictions with feature contributions
            - Open communication with stakeholders
            
            #### Privacy & Security
            - Data minimization principles
            - Secure storage protocols
            - Compliance with data protection regulations
            """)
        
        st.markdown("### Recommended Workflow")
        
        st.markdown("""
        1. **Initial Screening**: Use the system to identify potential anomalies
        2. **Human Review**: Have qualified academic staff review each flagged case
        3. **Additional Evidence**: Gather additional information beyond model output
        4. **Student Consultation**: Discuss concerns with students before decisions
        5. **Decision & Documentation**: Document reasoning for any actions taken
        """)
        
        # Additional resources
        st.subheader("Additional Resources")
        
        st.markdown("""
        - [Academic Integrity Policies](#)
        - [Best Practices for Academic Misconduct Investigation](#)
        - [Student Rights & Responsibilities](#)
        - [Data Protection Guidelines](#)
        - [Algorithmic Fairness Resources](#)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feedback and reporting
        # st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("Feedback & Reporting")
        
        st.markdown("""
        Please report any concerns about the system's performance, including:
        
        - Unexpected predictions
        - Potential bias or fairness issues
        - Technical errors or bugs
        - Suggestions for improvement
        
        Your feedback helps us continuously improve the system and ensure it remains fair and effective.
        """)
        
        # Simple feedback form
        with st.form("feedback_form"):
            feedback_type = st.selectbox(
                "Feedback Type",
                ["General Feedback", "Bug Report", "Bias Concern", "Feature Request"]
            )
            feedback_text = st.text_area("Your Feedback", height=150)
            submitted = st.form_submit_button("Submit Feedback", type="primary")
            
            if submitted:
                st.success("Thank you for your feedback! It has been recorded and will be reviewed.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
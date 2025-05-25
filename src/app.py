import streamlit as st
import joblib
from components.individual_analysis import render_individual_analysis
from components.batch_analysis import render_batch_analysis
from components.model_info import render_model_info
from components.ethical_guidelines import render_ethical_guidelines

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

# Load custom CSS
with open('src/styles/main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load model resources
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('cheating_detection_model.pkl')
        scaler = joblib.load('scaler.joblib')
        threshold = joblib.load('best_threshold.joblib')
        return model, scaler, threshold
    except FileNotFoundError:
        st.warning("Model files not found. Using simulated model for demonstration.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        scaler = None
        threshold = 0.7
        return model, scaler, threshold

# Define feature names
feature_names = ['coursework_z', 'exam_z', 'z_diff', 'score_variance', 
                 'exam_time_std', 'peer_comparison', 'subject_variation', 
                 'historical_trend', 'anomaly_score']

def main():
    # Load model resources
    model, scaler, threshold = load_model_resources()
    
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
        st.markdown("---")
        st.markdown("### System Status")
        st.success("✅ Model loaded successfully")
        st.success(f"✅ Detection threshold: {threshold:.3f}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Individual Analysis", "👥 Batch Analysis", "ℹ️ Model Information", "⚠️ Ethical Guidelines"])
    
    with tab1:
        render_individual_analysis(model, scaler, threshold, feature_names)
    
    with tab2:
        render_batch_analysis(model, scaler, threshold, feature_names)
    
    with tab3:
        render_model_info(model, feature_names)
    
    with tab4:
        render_ethical_guidelines()

if __name__ == "__main__":
    main()
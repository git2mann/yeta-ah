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

with open("src/styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    
    # Notion-style Hero section with accent bar
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom: 2.5rem;">
        <span class="notion-accent-bar"></span>
        <div>
            <h1 style="font-size: 2.7rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -0.01em;">🔍 Exam Anomaly Detection System</h1>
            <p style="font-size: 1.25rem; color: #a1a1aa; margin-bottom: 1.5rem;">An advanced machine learning solution for identifying potential academic misconduct patterns in exam performance.</p>
            <a href="#📊 Individual Analysis" class="btn-primary" style="margin-top: 1rem;">Get Started</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Notion-style callout/info box
    st.markdown("""
    <div class="notion-callout">
        <span style="font-size:1.5rem;">💡</span>
        <div>
            <strong>Tip:</strong> Try uploading your own CSV or use the demo data to explore the system's capabilities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights (Notion-style cards, add badge)
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; justify-content: center; margin-bottom: 2.5rem;">
        <div class="feature-card" style="flex:1; min-width:220px; text-align: center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
            <h3 style="margin: 0.5rem 0 0.5rem 0;">Individual Analysis <span class="notion-badge">New</span></h3>
            <p style="color:#a1a1aa;">Analyze students with detailed visualizations and feature breakdowns.</p>
        </div>
        <div class="feature-card" style="flex:1; min-width:220px; text-align: center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">👥</div>
            <h3 style="margin: 0.5rem 0 0.5rem 0;">Batch Analysis</h3>
            <p style="color:#a1a1aa;">Process multiple students at once and identify anomalies efficiently.</p>
        </div>
        <div class="feature-card" style="flex:1; min-width:220px; text-align: center;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📈</div>
            <h3 style="margin: 0.5rem 0 0.5rem 0;">Interactive Visuals</h3>
            <p style="color:#a1a1aa;">View radar charts, feature importances, and probability distributions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Notion-style divider
    st.markdown('<hr class="notion-divider">', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("IMG_7908.PNG", width=120)
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

    # Notion-style footer
    st.markdown("""
    <footer>
        <hr>
        <div>
            <span>© 2025 Exam Anomaly Detection System &nbsp;|&nbsp; <a href="https://github.com/yourusername/exam-anomaly-detection" style="color:#60a5fa;">GitHub</a></span>
        </div>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
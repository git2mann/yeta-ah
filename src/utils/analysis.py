import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def analyze_student(student_data, model, feature_names, threshold):
    """Analyzes student data and provides prediction with feature contributions"""
    # Ensure student_data is a NumPy array
    student_data = np.array(student_data)

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

def create_radar_chart(student_data, feature_names):
    """Creates an interactive radar chart for student features"""
    import numpy as np

    # Ensure student_data is a NumPy array
    student_data = np.array(student_data)

    fig = go.Figure()
    
    # Complete the loop for the radar chart
    values = student_data.tolist()  # Convert to list after ensuring it's a NumPy array
    values.append(values[0])  # Close the loop for the radar chart
    feature_names_loop = feature_names.copy()
    feature_names_loop.append(feature_names[0])  # Close the loop for feature names
    
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
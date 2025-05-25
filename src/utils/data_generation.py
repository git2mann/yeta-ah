import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def generate_demo_data(n_samples=100):
    """
    Generates synthetic demo data for the anomaly detection system.
    """
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
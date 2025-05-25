def calculate_metrics(coursework_scores, exam_scores, historical_scores, exam_times, peer_group_scores):
    """Calculate metrics from raw input data"""
    import numpy as np
    
    # Calculate z-scores for coursework and exam scores
    coursework_mean, coursework_std = np.mean(coursework_scores), np.std(coursework_scores)
    exam_mean, exam_std = np.mean(exam_scores), np.std(exam_scores)
    
    coursework_z = (coursework_scores[-1] - coursework_mean) / coursework_std
    exam_z = (exam_scores[-1] - exam_mean) / exam_std
    z_diff = coursework_z - exam_z

    # Calculate score variance
    score_variance = np.std(coursework_scores + exam_scores)

    # Calculate historical trend
    historical_trend = (coursework_scores[-1] + exam_scores[-1]) / 2 - np.mean(historical_scores)

    # Calculate standardized exam time
    exam_time_std = (exam_times[-1] - np.mean(exam_times)) / np.std(exam_times)

    # Calculate peer comparison
    peer_comparison = exam_scores[-1] - np.mean(peer_group_scores)

    # Calculate subject variation
    subject_variation = np.var(coursework_scores + exam_scores)

    return {
        'coursework_z': coursework_z,
        'exam_z': exam_z,
        'z_diff': z_diff,
        'score_variance': score_variance,
        'exam_time_std': exam_time_std,
        'peer_comparison': peer_comparison,
        'subject_variation': subject_variation,
        'historical_trend': historical_trend,
        'anomaly_score': 0.0  # Placeholder, calculated by model
    }
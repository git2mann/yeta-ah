def get_feature_explanation(feature):
    """Returns explanation and icon for a given feature"""
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
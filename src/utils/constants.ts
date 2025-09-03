import { FeatureInfo } from '../types';

export const FEATURE_NAMES = [
  'coursework_z',
  'exam_z', 
  'z_diff',
  'score_variance',
  'exam_time_std',
  'peer_comparison',
  'subject_variation',
  'historical_trend',
  'anomaly_score'
];

export const FEATURE_INFO: Record<string, FeatureInfo> = {
  coursework_z: {
    icon: '📝',
    title: 'Coursework Z-Score',
    description: 'Standardized coursework score. Unusually high scores may indicate plagiarism.'
  },
  exam_z: {
    icon: '📄',
    title: 'Exam Z-Score', 
    description: 'Standardized exam score. Unusually high scores may indicate advance knowledge of questions.'
  },
  z_diff: {
    icon: '⚖️',
    title: 'Z-Score Difference',
    description: 'Difference between coursework and exam z-scores. Large differences may indicate inconsistent knowledge.'
  },
  score_variance: {
    icon: '📊',
    title: 'Score Variance',
    description: 'Variance in student\'s scores across assessments. Low variance can indicate suspicious consistency.'
  },
  exam_time_std: {
    icon: '⏱️',
    title: 'Exam Time (standardized)',
    description: 'Standardized time taken to complete the exam. Very fast or slow completion times may be suspicious.'
  },
  peer_comparison: {
    icon: '👥',
    title: 'Peer Comparison',
    description: 'Performance relative to peer group. Significant outperformance of peers may be suspicious.'
  },
  subject_variation: {
    icon: '📚',
    title: 'Subject Variation',
    description: 'Variation in performance across subjects. Low variation can indicate suspicious consistency.'
  },
  historical_trend: {
    icon: '📈',
    title: 'Historical Trend',
    description: 'Change in performance compared to past. Sudden improvements may be suspicious.'
  },
  anomaly_score: {
    icon: '🔍',
    title: 'Anomaly Score',
    description: 'Statistical measure of how unusual the pattern is based on Isolation Forest algorithm.'
  }
};
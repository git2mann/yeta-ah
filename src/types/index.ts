export interface StudentData {
  coursework_z: number;
  exam_z: number;
  z_diff: number;
  score_variance: number;
  exam_time_std: number;
  peer_comparison: number;
  subject_variation: number;
  historical_trend: number;
  anomaly_score: number;
  student_id?: string;
}

export interface AnalysisResult {
  prediction: number;
  probability: number;
  feature_contributions: Record<string, number>;
}

export interface BatchResult {
  student_id: string;
  probability: number;
  prediction: number;
  flag: string;
}

export interface ModelInfo {
  model_type: string;
  feature_importances: Record<string, number>;
  threshold: number;
  performance_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
  };
}

export interface FeatureInfo {
  icon: string;
  title: string;
  description: string;
}
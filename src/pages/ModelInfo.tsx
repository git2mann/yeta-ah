import React, { useState, useEffect } from 'react';
import { getModelInfo } from '../utils/api';
import { ModelInfo as ModelInfoType } from '../types';
import { FEATURE_NAMES, FEATURE_INFO } from '../utils/constants';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Bot, TrendingUp, Target, Zap } from 'lucide-react';

const ModelInfo: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfoType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (error) {
      // Generate mock model info for demo
      const mockInfo: ModelInfoType = {
        model_type: 'Random Forest Classifier',
        feature_importances: Object.fromEntries(
          FEATURE_NAMES.map((name, i) => [name, Math.random() * 0.2 + 0.05])
        ),
        threshold: 0.7,
        performance_metrics: {
          accuracy: 0.92,
          precision: 0.85,
          recall: 0.78,
          f1_score: 0.81,
          auc_roc: 0.94
        }
      };
      setModelInfo(mockInfo);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">Failed to load model information.</p>
      </div>
    );
  }

  const featureImportanceData = Object.entries(modelInfo.feature_importances)
    .map(([feature, importance]) => ({
      feature: FEATURE_INFO[feature]?.title || feature,
      importance
    }))
    .sort((a, b) => b.importance - a.importance);

  // Generate ROC curve data for demo
  const rocData = Array.from({ length: 100 }, (_, i) => {
    const fpr = i / 99;
    const tpr = 1 / (1 + Math.exp(-9 * (fpr - 0.5)));
    return { fpr, tpr };
  });

  return (
    <div className="space-y-6">
      {/* Model Overview */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center mb-6">
          <div className="bg-primary-600 text-white rounded-full w-12 h-12 flex items-center justify-center mr-4">
            <Bot className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{modelInfo.model_type}</h2>
            <p className="text-gray-600">Optimized for detecting academic misconduct patterns</p>
          </div>
        </div>

        <div className="prose max-w-none">
          <h3 className="text-lg font-semibold mb-3">How It Works</h3>
          <p className="text-gray-600 mb-4">
            This model uses a supervised machine learning approach to detect potential academic misconduct:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-600">
            <li><strong>Data Collection:</strong> Gathers data about student performance in various assessments</li>
            <li><strong>Feature Engineering:</strong> Transforms raw data into relevant features</li>
            <li><strong>Training:</strong> Uses known cases to learn patterns of academic misconduct</li>
            <li><strong>Prediction:</strong> Applies learned patterns to identify anomalies in new data</li>
            <li><strong>Threshold Selection:</strong> Optimizes decision threshold to balance false positives and negatives</li>
          </ol>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <Target className="h-8 w-8 text-blue-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {(modelInfo.performance_metrics.accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Accuracy</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <Zap className="h-8 w-8 text-green-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {(modelInfo.performance_metrics.precision * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Precision</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <TrendingUp className="h-8 w-8 text-purple-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {(modelInfo.performance_metrics.recall * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Recall</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {modelInfo.performance_metrics.f1_score.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">F1 Score</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-2xl font-bold text-gray-900 mb-1">
            {modelInfo.performance_metrics.auc_roc.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">AUC-ROC</div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={featureImportanceData}
              layout="horizontal"
              margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={100} />
              <Tooltip 
                formatter={(value: number) => [value.toFixed(3), 'Importance']}
              />
              <Bar dataKey="importance" fill="#2563EB" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ROC Curve */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Model Performance - ROC Curve</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="fpr" 
                domain={[0, 1]}
                label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                domain={[0, 1]}
                label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value: number) => [value.toFixed(3)]}
                labelFormatter={(value: number) => `FPR: ${value.toFixed(3)}`}
              />
              <Line 
                type="monotone" 
                dataKey="tpr" 
                stroke="#2563EB" 
                strokeWidth={2}
                dot={false}
                name={`ROC Curve (AUC = ${modelInfo.performance_metrics.auc_roc.toFixed(3)})`}
              />
              <Line 
                type="monotone" 
                dataKey="fpr" 
                stroke="#9CA3AF" 
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Random Classifier"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Technical Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">Model Parameters</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• <strong>Model Type:</strong> Random Forest Classifier</li>
              <li>• <strong>Number of Trees:</strong> 100</li>
              <li>• <strong>Max Depth:</strong> 10</li>
              <li>• <strong>Feature Selection:</strong> All features</li>
              <li>• <strong>Cross-Validation:</strong> 5-fold</li>
              <li>• <strong>Metric Optimized:</strong> Precision-Recall AUC</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-3">Training Information</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• <strong>Training Data Size:</strong> 10,000 students</li>
              <li>• <strong>Positive Class:</strong> 12% (academic misconduct cases)</li>
              <li>• <strong>Training Period:</strong> Last 5 academic years</li>
              <li>• <strong>Data Sources:</strong> Anonymized academic records</li>
              <li>• <strong>Validation Method:</strong> Stratified K-fold</li>
              <li>• <strong>Feature Engineering:</strong> Z-score normalization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
import React, { useState } from 'react';
import { analyzeStudent, getDemoData } from '../utils/api';
import { StudentData, AnalysisResult } from '../types';
import { FEATURE_NAMES, FEATURE_INFO } from '../utils/constants';
import ProbabilityGauge from '../components/ProbabilityGauge';
import RadarChart from '../components/RadarChart';
import FeatureContributions from '../components/FeatureContributions';
import FeatureCard from '../components/FeatureCard';
import { AlertTriangle, CheckCircle, Play } from 'lucide-react';

const IndividualAnalysis: React.FC = () => {
  const [useDemo, setUseDemo] = useState(true);
  const [selectedStudent, setSelectedStudent] = useState<string>('');
  const [demoData, setDemoData] = useState<StudentData[]>([]);
  const [manualData, setManualData] = useState<StudentData>({
    coursework_z: 0,
    exam_z: 0,
    z_diff: 0,
    score_variance: 0,
    exam_time_std: 0,
    peer_comparison: 0,
    subject_variation: 0,
    historical_trend: 0,
    anomaly_score: 0
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  React.useEffect(() => {
    if (useDemo) {
      loadDemoData();
    }
  }, [useDemo]);

  const loadDemoData = async () => {
    try {
      const data = await getDemoData();
      setDemoData(data);
      if (data.length > 0) {
        setSelectedStudent(data[0].student_id || '0');
      }
    } catch (error) {
      // Generate demo data locally if API fails
      const mockData: StudentData[] = Array.from({ length: 20 }, (_, i) => ({
        student_id: `S${1000 + i}`,
        coursework_z: Math.random() * 4 - 2,
        exam_z: Math.random() * 4 - 2,
        z_diff: Math.random() * 2 - 1,
        score_variance: Math.random() * 2,
        exam_time_std: Math.random() * 4 - 2,
        peer_comparison: Math.random() * 2 - 1,
        subject_variation: Math.random() * 2,
        historical_trend: Math.random() * 2 - 1,
        anomaly_score: Math.random()
      }));
      setDemoData(mockData);
      setSelectedStudent(mockData[0].student_id || '0');
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      let studentData: StudentData;
      
      if (useDemo) {
        const student = demoData.find(s => s.student_id === selectedStudent);
        if (!student) return;
        studentData = student;
      } else {
        studentData = manualData;
      }

      const analysisResult = await analyzeStudent(studentData);
      setResult(analysisResult);
    } catch (error) {
      // Generate mock result for demo
      const studentData = useDemo 
        ? demoData.find(s => s.student_id === selectedStudent)!
        : manualData;
      
      const mockResult: AnalysisResult = {
        prediction: Math.random() > 0.7 ? 1 : 0,
        probability: Math.random() * 0.6 + 0.2,
        feature_contributions: Object.fromEntries(
          FEATURE_NAMES.map(name => [
            name, 
            (studentData[name as keyof StudentData] as number) * (Math.random() * 0.2 + 0.1)
          ])
        )
      };
      setResult(mockResult);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentStudentData = (): number[] => {
    if (useDemo) {
      const student = demoData.find(s => s.student_id === selectedStudent);
      return student ? FEATURE_NAMES.map(name => student[name as keyof StudentData] as number) : [];
    }
    return FEATURE_NAMES.map(name => manualData[name as keyof StudentData] as number);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Individual Student Analysis</h2>
        <p className="text-gray-600 mb-6">
          Analyze individual students with detailed visualizations and feature contributions.
        </p>

        {/* Data Source Selection */}
        <div className="mb-6">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={useDemo}
              onChange={(e) => setUseDemo(e.target.checked)}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm font-medium text-gray-700">Use demo data</span>
          </label>
        </div>

        {useDemo ? (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Student ID:
            </label>
            <select
              value={selectedStudent}
              onChange={(e) => setSelectedStudent(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            >
              {demoData.map((student) => (
                <option key={student.student_id} value={student.student_id}>
                  {student.student_id}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {FEATURE_NAMES.map((feature) => (
              <div key={feature}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {FEATURE_INFO[feature].title}
                </label>
                <input
                  type="number"
                  step="0.001"
                  value={manualData[feature as keyof StudentData]}
                  onChange={(e) => setManualData(prev => ({
                    ...prev,
                    [feature]: parseFloat(e.target.value) || 0
                  }))}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
            ))}
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={loading || (useDemo && !selectedStudent)}
          className="btn-primary flex items-center"
        >
          <Play className="mr-2 h-4 w-4" />
          {loading ? 'Analyzing...' : 'Analyze Student'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Prediction Results */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">Prediction Result</h3>
            
            <ProbabilityGauge probability={result.probability} threshold={0.7} />
            
            <div className={`mt-6 p-4 rounded-lg ${
              result.prediction === 1 
                ? 'bg-red-50 border-l-4 border-red-500' 
                : 'bg-green-50 border-l-4 border-green-500'
            }`}>
              <div className="flex items-center">
                {result.prediction === 1 ? (
                  <AlertTriangle className="h-6 w-6 text-red-600 mr-3" />
                ) : (
                  <CheckCircle className="h-6 w-6 text-green-600 mr-3" />
                )}
                <div>
                  <h4 className="font-bold text-lg">
                    {result.prediction === 1 
                      ? 'Potential Academic Misconduct Detected' 
                      : 'No Anomaly Detected'
                    }
                  </h4>
                  <p className="text-sm mt-1">
                    {result.prediction === 1
                      ? 'This student\'s performance pattern shows anomalies that warrant further investigation.'
                      : 'This student\'s performance pattern appears consistent with expected behavior.'
                    }
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <h4 className="font-bold mb-3">Top Contributing Factors</h4>
              <FeatureContributions contributions={result.feature_contributions} />
            </div>
          </div>

          {/* Feature Visualization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">Feature Visualization</h3>
            
            <div className="mb-6">
              <RadarChart studentData={getCurrentStudentData()} />
            </div>

            <h4 className="font-bold mb-3">Key Features</h4>
            <div className="space-y-3">
              {Object.entries(result.feature_contributions)
                .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                .slice(0, 3)
                .map(([feature, contribution]) => {
                  const studentData = getCurrentStudentData();
                  const featureIndex = FEATURE_NAMES.indexOf(feature);
                  const value = featureIndex >= 0 ? studentData[featureIndex] : 0;
                  
                  return (
                    <FeatureCard
                      key={feature}
                      feature={feature}
                      value={value}
                      contribution={contribution}
                    />
                  );
                })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndividualAnalysis;
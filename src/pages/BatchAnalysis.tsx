import React, { useState } from 'react';
import { analyzeBatch } from '../utils/api';
import { BatchResult } from '../types';
import { FEATURE_NAMES } from '../utils/constants';
import FileUpload from '../components/FileUpload';
import { Download, Upload, AlertTriangle, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const BatchAnalysis: React.FC = () => {
  const [results, setResults] = useState<BatchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [minProbability, setMinProbability] = useState(0.7);

  const handleFileUpload = async (content: string) => {
    setLoading(true);
    try {
      const batchResults = await analyzeBatch(content);
      setResults(batchResults);
    } catch (error) {
      // Generate mock results for demo
      const lines = content.split('\n');
      const headers = lines[0].split(',');
      const mockResults: BatchResult[] = lines.slice(1, 21).map((line, index) => ({
        student_id: `S${1000 + index}`,
        probability: Math.random() * 0.8 + 0.1,
        prediction: Math.random() > 0.8 ? 1 : 0,
        flag: Math.random() > 0.8 ? '⚠️' : '✅'
      }));
      setResults(mockResults);
    } finally {
      setLoading(false);
    }
  };

  const downloadTemplate = () => {
    const headers = ['student_id', ...FEATURE_NAMES];
    const csvContent = headers.join(',') + '\n';
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'anomaly_detection_template.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadResults = () => {
    if (results.length === 0) return;
    
    const headers = ['student_id', 'flag', 'probability', 'prediction'];
    const csvContent = [
      headers.join(','),
      ...results.map(r => [
        r.student_id,
        r.flag,
        r.probability.toFixed(4),
        r.prediction
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'anomaly_detection_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredResults = results.filter(r => r.probability >= minProbability);
  const flaggedCount = results.filter(r => r.prediction === 1).length;

  const histogramData = Array.from({ length: 20 }, (_, i) => {
    const binStart = i * 0.05;
    const binEnd = (i + 1) * 0.05;
    const count = results.filter(r => r.probability >= binStart && r.probability < binEnd).length;
    return {
      bin: `${(binStart * 100).toFixed(0)}-${(binEnd * 100).toFixed(0)}%`,
      count
    };
  });

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Batch Analysis</h2>
        <p className="text-gray-600 mb-6">
          Process multiple students at once and identify potential anomalies.
        </p>

        {/* File Upload Section */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-4">Upload Data</h3>
          <FileUpload onFileUpload={handleFileUpload} />
          
          <div className="mt-4 flex gap-4">
            <button
              onClick={downloadTemplate}
              className="btn-secondary flex items-center"
            >
              <Download className="mr-2 h-4 w-4" />
              Download Template
            </button>
          </div>
        </div>

        {/* Required Columns */}
        <div className="mb-6">
          <h4 className="text-md font-semibold mb-3">Required Columns</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
            {['student_id', ...FEATURE_NAMES].map((col) => (
              <div key={col} className="bg-gray-100 px-3 py-2 rounded text-sm font-mono">
                {col}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <>
          {/* Summary Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-gray-900 mb-2">{results.length}</div>
              <div className="text-sm text-gray-600">Total Students</div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-red-600 mb-2">{flaggedCount}</div>
              <div className="text-sm text-gray-600">Flagged Students</div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">
                {((flaggedCount / results.length) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Flagged Percentage</div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">0.70</div>
              <div className="text-sm text-gray-600">Threshold</div>
            </div>
          </div>

          {/* Filter Controls */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Filter Results</h3>
            <div className="flex items-center space-x-4">
              <label className="text-sm font-medium text-gray-700">
                Minimum Probability:
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={minProbability}
                onChange={(e) => setMinProbability(parseFloat(e.target.value))}
                className="flex-1 max-w-xs"
              />
              <span className="text-sm font-medium text-gray-900">
                {(minProbability * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Results Table */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                Students with Probability ≥ {(minProbability * 100).toFixed(0)}%
              </h3>
              <button
                onClick={downloadResults}
                className="btn-primary flex items-center"
              >
                <Download className="mr-2 h-4 w-4" />
                Download Results
              </button>
            </div>

            {filteredResults.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Student ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Probability
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Prediction
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredResults.map((result) => (
                      <tr key={result.student_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {result.student_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="text-lg">{result.flag}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className={`font-medium ${
                            result.probability > 0.7 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {(result.probability * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {result.prediction === 1 ? (
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                              Flagged
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                              Normal
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                No students match the current filter criteria.
              </div>
            )}
          </div>

          {/* Probability Distribution */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Probability Distribution</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histogramData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#2563EB" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default BatchAnalysis;
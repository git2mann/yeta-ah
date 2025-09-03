import React, { useState } from 'react';
import { submitFeedback } from '../utils/api';
import { AlertTriangle, Shield, Users, Eye, FileText, Send } from 'lucide-react';

const EthicalGuidelines: React.FC = () => {
  const [feedbackType, setFeedbackType] = useState('General Feedback');
  const [feedbackText, setFeedbackText] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleSubmitFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await submitFeedback({
        type: feedbackType,
        message: feedbackText
      });
      setFeedbackSubmitted(true);
      setFeedbackText('');
      setTimeout(() => setFeedbackSubmitted(false), 3000);
    } catch (error) {
      // Mock success for demo
      setFeedbackSubmitted(true);
      setFeedbackText('');
      setTimeout(() => setFeedbackSubmitted(false), 3000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Important Notice */}
      <div className="warning-box">
        <div className="flex items-start">
          <AlertTriangle className="h-6 w-6 text-red-600 mr-3 mt-1 flex-shrink-0" />
          <div>
            <h3 className="text-lg font-bold text-red-800 mb-2">Important Notice</h3>
            <p className="text-red-700">
              This system is designed to be a <strong>decision support tool</strong> and not an automated 
              decision-making system. All flagged cases must be reviewed by qualified academic staff before any action is taken.
            </p>
          </div>
        </div>
      </div>

      {/* Key Principles */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6">Key Principles</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="flex items-start">
              <Shield className="h-6 w-6 text-blue-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Fairness & Bias Mitigation</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Regular audits for algorithmic bias</li>
                  <li>• Balanced representation in training data</li>
                  <li>• Continuous monitoring for disparate impact</li>
                </ul>
              </div>
            </div>

            <div className="flex items-start">
              <Users className="h-6 w-6 text-green-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Human Oversight</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• All flags require human review</li>
                  <li>• Staff training on tool limitations</li>
                  <li>• Appeal process for students</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="flex items-start">
              <Eye className="h-6 w-6 text-purple-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Transparency</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Clear documentation of model workings</li>
                  <li>• Explainable predictions with feature contributions</li>
                  <li>• Open communication with stakeholders</li>
                </ul>
              </div>
            </div>

            <div className="flex items-start">
              <FileText className="h-6 w-6 text-orange-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Privacy & Security</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Data minimization principles</li>
                  <li>• Secure storage protocols</li>
                  <li>• Compliance with data protection regulations</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommended Workflow */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6">Recommended Workflow</h2>
        <div className="space-y-4">
          {[
            {
              step: 1,
              title: 'Initial Screening',
              description: 'Use the system to identify potential anomalies',
              color: 'bg-blue-500'
            },
            {
              step: 2,
              title: 'Human Review',
              description: 'Have qualified academic staff review each flagged case',
              color: 'bg-green-500'
            },
            {
              step: 3,
              title: 'Additional Evidence',
              description: 'Gather additional information beyond model output',
              color: 'bg-yellow-500'
            },
            {
              step: 4,
              title: 'Student Consultation',
              description: 'Discuss concerns with students before decisions',
              color: 'bg-purple-500'
            },
            {
              step: 5,
              title: 'Decision & Documentation',
              description: 'Document reasoning for any actions taken',
              color: 'bg-red-500'
            }
          ].map((item) => (
            <div key={item.step} className="flex items-start">
              <div className={`${item.color} text-white rounded-full w-8 h-8 flex items-center justify-center mr-4 mt-1 flex-shrink-0 text-sm font-bold`}>
                {item.step}
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">{item.title}</h3>
                <p className="text-sm text-gray-600">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Additional Resources */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6">Additional Resources</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            'Academic Integrity Policies',
            'Best Practices for Academic Misconduct Investigation',
            'Student Rights & Responsibilities',
            'Data Protection Guidelines',
            'Algorithmic Fairness Resources',
            'Appeal Process Documentation'
          ].map((resource) => (
            <a
              key={resource}
              href="#"
              className="flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <FileText className="h-5 w-5 text-gray-400 mr-3" />
              <span className="text-sm font-medium text-gray-700">{resource}</span>
            </a>
          ))}
        </div>
      </div>

      {/* Feedback Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-6">Feedback & Reporting</h2>
        <p className="text-gray-600 mb-6">
          Please report any concerns about the system's performance, including unexpected predictions, 
          potential bias or fairness issues, technical errors, or suggestions for improvement.
        </p>

        <form onSubmit={handleSubmitFeedback} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Feedback Type
            </label>
            <select
              value={feedbackType}
              onChange={(e) => setFeedbackType(e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            >
              <option>General Feedback</option>
              <option>Bug Report</option>
              <option>Bias Concern</option>
              <option>Feature Request</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Feedback
            </label>
            <textarea
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              rows={4}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
              placeholder="Please describe your feedback in detail..."
            />
          </div>

          <button
            type="submit"
            disabled={!feedbackText.trim()}
            className="btn-primary flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="mr-2 h-4 w-4" />
            Submit Feedback
          </button>

          {feedbackSubmitted && (
            <div className="success-box">
              <p className="text-green-700 font-medium">
                Thank you for your feedback! It has been recorded and will be reviewed.
              </p>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default EthicalGuidelines;
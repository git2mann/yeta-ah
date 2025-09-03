import React from 'react';
import { Link } from 'react-router-dom';
import { User, Users, Info, Shield, BarChart3, ArrowRight } from 'lucide-react';

const Dashboard: React.FC = () => {
  const features = [
    {
      icon: User,
      title: 'Individual Student Analysis',
      description: 'Analyze individual students with detailed visualizations including radar charts, feature contributions, and probability assessments.',
      href: '/individual',
      color: 'bg-blue-500'
    },
    {
      icon: Users,
      title: 'Batch Analysis',
      description: 'Process multiple students at once, identify potential anomalies, and export results for further investigation.',
      href: '/batch',
      color: 'bg-green-500'
    },
    {
      icon: BarChart3,
      title: 'Interactive Visualizations',
      description: 'Explore data through interactive charts, including radar plots, feature contributions, and probability distributions.',
      href: '/individual',
      color: 'bg-purple-500'
    },
    {
      icon: Info,
      title: 'Model Information',
      description: 'Understand the model\'s methodology, feature importance, and the underlying patterns it detects.',
      href: '/model',
      color: 'bg-orange-500'
    },
    {
      icon: Shield,
      title: 'Ethical Considerations',
      description: 'Built with ethical guidelines in mind, emphasizing human oversight and fair application of results.',
      href: '/ethics',
      color: 'bg-red-500'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-900 rounded-xl p-8 text-white">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Exam Anomaly Detection System</h1>
          <p className="text-xl mb-8 max-w-3xl mx-auto opacity-90">
            An advanced machine learning solution for identifying potential academic misconduct patterns in exam performance.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link to="/individual" className="btn-primary bg-white text-primary-600 hover:bg-gray-100">
              Get Started
            </Link>
            <Link to="/model" className="btn-secondary border-white text-white hover:bg-white hover:text-primary-600">
              Learn More
            </Link>
          </div>
        </div>
      </div>

      {/* About Section */}
      <div className="bg-white rounded-lg shadow-md p-8">
        <h2 className="text-3xl font-bold mb-6 text-center text-gray-800">About This Application</h2>
        <div className="prose max-w-none">
          <p className="text-lg mb-4">
            This application provides an intuitive interface for detecting potential exam anomalies using machine learning. 
            The system analyzes student performance patterns across multiple dimensions to identify potential cases of academic misconduct.
          </p>
          <p className="text-lg mb-4">
            By combining statistical analysis with machine learning techniques, the system can detect subtle patterns that might indicate 
            cheating, such as inconsistent performance between coursework and exams, unusual timing patterns, or significant deviations from peer performance.
          </p>
          <div className="info-box">
            <p className="font-medium">
              <strong>Note:</strong> This system is designed as a screening tool to identify potential cases for further review, 
              not as a definitive determination of academic misconduct.
            </p>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <Link
              key={index}
              to={feature.href}
              className="feature-card group"
            >
              <div className={`${feature.color} w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <Icon className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-900">{feature.title}</h3>
              <p className="text-gray-600 mb-4">{feature.description}</p>
              <div className="flex items-center text-primary-600 font-medium group-hover:text-primary-700">
                Explore <ArrowRight className="ml-2 h-4 w-4" />
              </div>
            </Link>
          );
        })}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-3xl font-bold text-primary-600 mb-2">92%</div>
          <div className="text-sm text-gray-600">Model Accuracy</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-3xl font-bold text-green-600 mb-2">85%</div>
          <div className="text-sm text-gray-600">Precision</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-3xl font-bold text-blue-600 mb-2">78%</div>
          <div className="text-sm text-gray-600">Recall</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-3xl font-bold text-purple-600 mb-2">0.94</div>
          <div className="text-sm text-gray-600">AUC-ROC</div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
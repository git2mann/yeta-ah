import React from 'react';
import { FEATURE_INFO } from '../utils/constants';

interface FeatureCardProps {
  feature: string;
  value: number;
  contribution?: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ feature, value, contribution }) => {
  const info = FEATURE_INFO[feature];
  
  return (
    <div className="bg-gray-50 p-4 rounded-lg border">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="text-sm font-medium text-gray-900 flex items-center">
            <span className="mr-2">{info.icon}</span>
            {info.title}
          </h4>
          <p className="text-xs text-gray-600 mt-1">{info.description}</p>
          <div className="mt-2 space-y-1">
            <p className="text-sm">
              <span className="font-medium">Value:</span> {value.toFixed(3)}
            </p>
            {contribution !== undefined && (
              <p className="text-sm">
                <span className="font-medium">Contribution:</span> {contribution.toFixed(3)}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureCard;
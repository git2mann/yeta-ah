import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface FeatureContributionsProps {
  contributions: Record<string, number>;
  topN?: number;
}

const FeatureContributions: React.FC<FeatureContributionsProps> = ({ 
  contributions, 
  topN = 5 
}) => {
  const data = Object.entries(contributions)
    .map(([feature, contribution]) => ({
      feature,
      contribution,
      absolute: Math.abs(contribution)
    }))
    .sort((a, b) => b.absolute - a.absolute)
    .slice(0, topN);

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="horizontal"
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="feature" type="category" width={80} />
          <Tooltip 
            formatter={(value: number) => [value.toFixed(3), 'Contribution']}
            labelStyle={{ color: '#374151' }}
          />
          <Bar 
            dataKey="contribution" 
            fill={(entry) => entry.contribution > 0 ? '#EF4444' : '#22C55E'}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureContributions;
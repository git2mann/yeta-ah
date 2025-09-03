import React from 'react';
import Plot from 'react-plotly.js';
import { FEATURE_NAMES } from '../utils/constants';

interface RadarChartProps {
  studentData: number[];
  featureNames?: string[];
}

const RadarChart: React.FC<RadarChartProps> = ({ 
  studentData, 
  featureNames = FEATURE_NAMES 
}) => {
  // Complete the loop for the radar chart
  const values = [...studentData, studentData[0]];
  const features = [...featureNames, featureNames[0]];

  const data = [{
    type: 'scatterpolar',
    r: values,
    theta: features,
    fill: 'toself',
    fillcolor: 'rgba(37, 99, 235, 0.2)',
    line: { color: '#2563EB', width: 2 },
    name: 'Student Features'
  }];

  const layout = {
    polar: {
      radialaxis: {
        visible: true,
        range: [-3, 3]
      }
    },
    showlegend: false,
    width: 400,
    height: 400,
    margin: { l: 80, r: 80, t: 20, b: 20 },
    font: { family: 'Inter, sans-serif' }
  };

  return (
    <Plot
      data={data}
      layout={layout}
      style={{ width: '100%', height: '400px' }}
      config={{ displayModeBar: false }}
    />
  );
};

export default RadarChart;
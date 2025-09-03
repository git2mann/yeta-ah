import React from 'react';
import Plot from 'react-plotly.js';

interface ProbabilityGaugeProps {
  probability: number;
  threshold: number;
}

const ProbabilityGauge: React.FC<ProbabilityGaugeProps> = ({ probability, threshold }) => {
  const data = [{
    type: 'indicator',
    mode: 'gauge+number',
    value: probability,
    domain: { x: [0, 1], y: [0, 1] },
    title: { text: 'Probability of Academic Misconduct' },
    gauge: {
      axis: { range: [0, 1], tickwidth: 1, tickcolor: 'darkblue' },
      bar: { color: probability <= threshold ? '#2563EB' : '#EF4444' },
      bgcolor: 'white',
      borderwidth: 2,
      bordercolor: 'gray',
      steps: [
        { range: [0, threshold], color: '#DBEAFE' },
        { range: [threshold, 1], color: '#FEE2E2' }
      ],
      threshold: {
        line: { color: 'black', width: 4 },
        thickness: 0.75,
        value: threshold
      }
    }
  }];

  const layout = {
    height: 300,
    margin: { l: 20, r: 20, t: 50, b: 20 },
    font: { family: 'Inter, sans-serif' }
  };

  return (
    <Plot
      data={data}
      layout={layout}
      style={{ width: '100%', height: '300px' }}
      config={{ displayModeBar: false }}
    />
  );
};

export default ProbabilityGauge;
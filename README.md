# Exam Anomaly Detection System - React Frontend

A modern React.js frontend for the Exam Anomaly Detection System, providing an intuitive interface for detecting potential exam anomalies using machine learning.

## Features

- **Modern React Interface**: Built with React 18, TypeScript, and Tailwind CSS
- **Individual Student Analysis**: Analyze individual students with interactive visualizations
- **Batch Analysis**: Process multiple students via CSV upload with real-time results
- **Interactive Charts**: Radar charts, probability gauges, and feature contribution plots
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **FastAPI Backend**: High-performance Python backend with automatic API documentation

## Architecture

### Frontend (React)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom components
- **Charts**: Recharts and Plotly.js for interactive visualizations
- **Routing**: React Router for navigation
- **File Upload**: React Dropzone for CSV uploads
- **HTTP Client**: Axios for API communication

### Backend (FastAPI)
- **Framework**: FastAPI with automatic OpenAPI documentation
- **ML Model**: Scikit-learn Random Forest classifier
- **Data Processing**: Pandas and NumPy
- **CORS**: Enabled for frontend-backend communication

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ and pip

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to `http://localhost:3000`

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```bash
   python run.py
   ```

4. The API will be available at `http://localhost:8000`
5. View API documentation at `http://localhost:8000/docs`

## Project Structure

```
├── src/
│   ├── components/          # Reusable React components
│   │   ├── Layout.tsx       # Main layout with navigation
│   │   ├── FeatureCard.tsx  # Feature information cards
│   │   ├── ProbabilityGauge.tsx  # Probability visualization
│   │   ├── RadarChart.tsx   # Radar chart component
│   │   ├── FeatureContributions.tsx  # Feature contribution charts
│   │   └── FileUpload.tsx   # File upload component
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx    # Main dashboard
│   │   ├── IndividualAnalysis.tsx  # Individual student analysis
│   │   ├── BatchAnalysis.tsx       # Batch processing
│   │   ├── ModelInfo.tsx    # Model information and metrics
│   │   └── EthicalGuidelines.tsx   # Ethical guidelines and feedback
│   ├── types/               # TypeScript type definitions
│   ├── utils/               # Utility functions and API client
│   └── App.tsx             # Main application component
├── backend/
│   ├── main.py             # FastAPI application
│   ├── run.py              # Server startup script
│   └── requirements.txt    # Python dependencies
└── public/                 # Static assets
```

## API Endpoints

- `POST /analyze-student` - Analyze individual student data
- `POST /analyze-batch` - Analyze multiple students from CSV
- `GET /model-info` - Get model information and performance metrics
- `GET /demo-data` - Generate demo student data
- `POST /feedback` - Submit user feedback
- `GET /health` - Health check endpoint

## Data Format

For batch analysis, upload a CSV file with these columns:

| Column | Description |
|--------|-------------|
| `student_id` | Unique student identifier (optional) |
| `coursework_z` | Standardized coursework score |
| `exam_z` | Standardized exam score |
| `z_diff` | Difference between coursework and exam z-scores |
| `score_variance` | Variance in student's scores across assessments |
| `exam_time_std` | Standardized time taken to complete exam |
| `peer_comparison` | Performance relative to peer group |
| `subject_variation` | Variation in performance across subjects |
| `historical_trend` | Change in performance compared to past |
| `anomaly_score` | Statistical measure of pattern unusualness |

## Development

### Frontend Development
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend Development
```bash
cd backend
python run.py        # Start development server with auto-reload
```

## Deployment

### Frontend
The React app can be deployed to any static hosting service:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

### Backend
The FastAPI backend can be deployed to:
- Heroku
- AWS Lambda (with Mangum)
- Google Cloud Run
- DigitalOcean App Platform

## Important Notes

- This system is designed as a **screening tool** for identifying potential cases for further review
- All flagged cases should be reviewed by qualified academic staff
- The system should be used in accordance with institutional policies and ethical guidelines
- Regular audits should be conducted to ensure fairness and prevent bias

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
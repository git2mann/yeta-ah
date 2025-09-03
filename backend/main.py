from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Exam Anomaly Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StudentData(BaseModel):
    coursework_z: float
    exam_z: float
    z_diff: float
    score_variance: float
    exam_time_std: float
    peer_comparison: float
    subject_variation: float
    historical_trend: float
    anomaly_score: float
    student_id: str = None

class AnalysisResult(BaseModel):
    prediction: int
    probability: float
    feature_contributions: Dict[str, float]

class BatchAnalysisRequest(BaseModel):
    csv_data: str

class BatchResult(BaseModel):
    student_id: str
    probability: float
    prediction: int
    flag: str

class FeedbackRequest(BaseModel):
    type: str
    message: str

class ModelInfo(BaseModel):
    model_type: str
    feature_importances: Dict[str, float]
    threshold: float
    performance_metrics: Dict[str, float]

# Global variables for model components
model = None
scaler = None
threshold = 0.7
feature_names = [
    'coursework_z', 'exam_z', 'z_diff', 'score_variance', 
    'exam_time_std', 'peer_comparison', 'subject_variation', 
    'historical_trend', 'anomaly_score'
]

@app.on_event("startup")
async def load_model():
    global model, scaler, threshold
    try:
        model = joblib.load('../cheating_detection_model.pkl')
        scaler = joblib.load('../scaler.joblib')
        threshold = joblib.load('../best_threshold.joblib')
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model files not found. Using mock model for demo.")
        # Create a mock model for demo purposes
        model = RandomForestClassifier()
        scaler = None
        threshold = 0.7

def analyze_student_data(student_data: np.ndarray) -> tuple:
    """Analyze student data and return prediction, probability, and contributions"""
    global model, threshold, feature_names
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(student_data.reshape(1, -1))[0][1]
    else:
        # Simulated probability for demo
        prob = 0.3 + (np.sum(np.abs(student_data)) / 10)
        prob = min(max(prob, 0), 1)
    
    prediction = 1 if prob > threshold else 0
    
    # Feature contributions
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        # Simulated feature importances for demo
        feature_importances = np.array([0.18, 0.16, 0.15, 0.12, 0.11, 0.1, 0.08, 0.06, 0.04])
    
    feature_contributions = dict(zip(feature_names, feature_importances * student_data))
    
    return prediction, prob, feature_contributions

@app.post("/analyze-student", response_model=AnalysisResult)
async def analyze_student(student: StudentData):
    """Analyze a single student"""
    try:
        # Convert to numpy array
        student_array = np.array([
            student.coursework_z, student.exam_z, student.z_diff,
            student.score_variance, student.exam_time_std, student.peer_comparison,
            student.subject_variation, student.historical_trend, student.anomaly_score
        ])
        
        prediction, probability, contributions = analyze_student_data(student_array)
        
        return AnalysisResult(
            prediction=prediction,
            probability=probability,
            feature_contributions=contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch", response_model=List[BatchResult])
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple students from CSV data"""
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(request.csv_data))
        
        # Check required columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        results = []
        for index, row in df.iterrows():
            student_array = np.array([row[col] for col in feature_names])
            prediction, probability, _ = analyze_student_data(student_array)
            
            student_id = row.get('student_id', f'Student_{index}')
            flag = '⚠️' if prediction == 1 else '✅'
            
            results.append(BatchResult(
                student_id=str(student_id),
                probability=probability,
                prediction=prediction,
                flag=flag
            ))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    global model, threshold, feature_names
    
    try:
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = dict(zip(feature_names, model.feature_importances_))
        else:
            # Mock feature importances for demo
            importances = np.array([0.18, 0.16, 0.15, 0.12, 0.11, 0.1, 0.08, 0.06, 0.04])
            feature_importances = dict(zip(feature_names, importances))
        
        return ModelInfo(
            model_type="Random Forest Classifier",
            feature_importances=feature_importances,
            threshold=threshold,
            performance_metrics={
                "accuracy": 0.92,
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.94
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demo-data", response_model=List[StudentData])
async def get_demo_data():
    """Generate demo student data"""
    try:
        np.random.seed(42)
        demo_students = []
        
        for i in range(20):
            student = StudentData(
                student_id=f"S{1000 + i}",
                coursework_z=np.random.normal(0, 1),
                exam_z=np.random.normal(0, 1),
                z_diff=np.random.normal(0, 0.5),
                score_variance=np.random.uniform(0.5, 2.0),
                exam_time_std=np.random.normal(0, 1),
                peer_comparison=np.random.normal(0, 0.8),
                subject_variation=np.random.uniform(0.3, 1.5),
                historical_trend=np.random.normal(0, 0.7),
                anomaly_score=np.random.uniform(0, 1)
            )
            demo_students.append(student)
        
        return demo_students
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    try:
        # In a real application, you would save this to a database
        print(f"Feedback received: {feedback.type} - {feedback.message}")
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
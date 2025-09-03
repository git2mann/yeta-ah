#!/usr/bin/env python3
"""
Simple script to run the FastAPI backend server
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting Exam Anomaly Detection API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
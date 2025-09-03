import axios from 'axios';
import { StudentData, AnalysisResult, BatchResult, ModelInfo } from '../types';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeStudent = async (studentData: StudentData): Promise<AnalysisResult> => {
  const response = await api.post('/analyze-student', studentData);
  return response.data;
};

export const analyzeBatch = async (csvData: string): Promise<BatchResult[]> => {
  const response = await api.post('/analyze-batch', { csv_data: csvData });
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get('/model-info');
  return response.data;
};

export const getDemoData = async (): Promise<StudentData[]> => {
  const response = await api.get('/demo-data');
  return response.data;
};

export const submitFeedback = async (feedback: {
  type: string;
  message: string;
}): Promise<void> => {
  await api.post('/feedback', feedback);
};
# Exam Anomaly Detection System

This web application provides an interface for detecting potential exam anomalies using machine learning. The system analyzes student performance patterns to identify potential cases of academic misconduct.

## Features

- **Individual Student Analysis**: Analyze individual students with detailed visualizations
- **Batch Analysis**: Process multiple students at once and identify potential anomalies
- **Interactive Visualizations**: View radar charts, feature contributions, and probability distributions
- **Model Information**: Understand the model's methodology and feature importance

## Getting Started

### 1. Prerequisites

- **Python 3.8 or higher** is required. You can check your version with:
  ```
  python3 --version
  ```
- **pip** (Python package manager). If not available, install it with:
  ```
  python3 -m ensurepip --upgrade
  ```
- (Recommended) **Virtual environment** for isolated dependencies:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Install the required dependencies

- Install all Python dependencies:
  ```
  pip install -r requirements.txt
  ```
  If you have multiple Python versions, use:
  ```
  pip3 install -r requirements.txt
  ```

### 3. Add Python user scripts to your PATH (if needed)

If you installed packages with `--user` and get "command not found" for `streamlit`, add this to your `~/.zshrc` or `~/.bash_profile`:
  ```
  export PATH="$HOME/Library/Python/3.9/bin:$PATH"
  ```
Then reload your shell:
  ```
  source ~/.zshrc
  ```

### 4. Run the Streamlit application

- For the main app:
  ```
  streamlit run app.py
  ```
- For the modular version:
  ```
  streamlit run src/app.py
  ```
  or
  ```
  python3 -m streamlit run src/app.py
  ```

### 5. Access the application

Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## Data Requirements

For batch analysis, upload a CSV file with the following columns:
- coursework_z
- exam_z
- z_diff
- score_variance
- exam_time_std
- peer_comparison
- subject_variation
- historical_trend
- anomaly_score

---

## Troubleshooting

- If you see "command not found: streamlit", use:
  ```
  python3 -m streamlit run src/app.py
  ```
- If you have issues with dependencies, ensure your Python version matches the requirements and consider using a virtual environment.

---

## Model Information

The system uses a Random Forest classifier trained on patterns of academic performance. It identifies potential anomalies based on:

1. Inconsistent performance between coursework and exams
2. Unusual patterns across different assessments
3. Suspicious timing during exams
4. Sudden improvements compared to historical performance
5. Performance that significantly deviates from peer group

---

## Important Note

This system is designed as a screening tool to identify potential cases for further review, not as a definitive determination of academic misconduct. All flagged cases should be reviewed by academic staff following appropriate institutional policies.
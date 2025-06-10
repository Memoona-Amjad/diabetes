# DiabeteSmart Analyzer

A machine learning-powered web application for diabetes risk prediction and visualization.

**Live Demo:** [https://diabetesmart-analyzer.streamlit.app/](https://diabetesmart-analyzer.streamlit.app/)

## Project Overview

DiabeteSmart Analyzer is a comprehensive tool that allows users to assess their risk of diabetes based on various health metrics. The application uses a Random Forest classifier trained on the Pima Indians Diabetes Dataset to provide personalized risk assessments, visualize risk factors, and offer tailored recommendations.

### Key Features

- **Personalized Risk Assessment**: Calculate diabetes risk probability based on key health metrics
- **Interactive Visualizations**: View risk factors through intuitive radar charts and gauges
- **Educational Content**: Learn about diabetes, its causes, and prevention strategies
- **Feedback System**: Provide feedback and view analytics from other users
- **BMI Calculator**: Calculate BMI from height and weight measurements

## Repository Structure

```
diabetesmart-analyzer/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Required Python packages
├── README.md                   # This file
├── diabetes_prediction_model.pkl  # Random Forest classifier
├── diabetes_scaler.pkl         # StandardScaler for preprocessing
└── diabetes.csv                # Dataset for model training
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/memoona215083/diabetesmart-analyzer.git
   cd diabetesmart-analyzer
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Access the application in your web browser at:
   ```
   http://localhost:8501
   ```

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning model development
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Web application framework

## About the Author

This project was developed by Memoona Amjad (215083) as part of the Data Science course at BSCS-F21.

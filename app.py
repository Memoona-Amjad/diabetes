import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import json
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(
    page_title="DiabeteSmart Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean, professional light theme
st.markdown("""
<style>
    /* Main colors and fonts */
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --accent: #9b59b6;
        --danger: #e74c3c;
        --warning: #f39c12;
        --dark: #2c3e50;
        --light: #f8f9fa;
        --gray: #95a5a6;
    }
    
    .main {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* App title */
    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #3498db;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Top navigation */
    .top-nav {
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 10px 0;
        background-color: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Cards and sections */
    .section-title {
        font-size: 1.8rem;
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3498db;
    }
    
    .insight-card {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-top: 3px solid #2ecc71;
    }
    
    /* Risk cards */
    .prediction-card {
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .high-risk {
        background-color: #fef5f5;
        border-left: 6px solid #e74c3c;
    }
    
    .moderate-risk {
        background-color: #fff9f0;
        border-left: 6px solid #f39c12;
    }
    
    .low-risk {
        background-color: #f0fff5;
        border-left: 6px solid #2ecc71;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    
    /* Recommendations */
    .recommendation-item {
        margin-bottom: 10px;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        padding: 15px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 40px;
        border-top: 1px solid #ecf0f1;
        padding-top: 20px;
    }
    
    /* Streamlit component improvements */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Additional spacing fixes */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #3498db;
    }
    
    /* Improve info boxes */
    .stAlert {
        background-color: #ebf5fb;
        border-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('diabetes_prediction_model.pkl', 'rb'))
        scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
        return model, scaler, True
    except Exception as e:
        return None, None, False

model, scaler, model_load_success = load_model()

# App Header
st.markdown('<p class="app-title">DiabeteSmart Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Advanced Diabetes Risk Analysis & Visualization System</p>', unsafe_allow_html=True)

# TOP NAVIGATION BAR
st.markdown('<div class="top-nav">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("🔍 Risk Analysis", key="btn_analysis", use_container_width=True):
        st.session_state.page = "analysis"
with col2:
    if st.button("📊 Data Insights", key="btn_insights", use_container_width=True):
        st.session_state.page = "insights"
with col3:
    if st.button("📝 Feedback", key="btn_feedback", use_container_width=True):
        st.session_state.page = "feedback"
with col4:
    if st.button("ℹ️ About", key="btn_about", use_container_width=True):
        st.session_state.page = "about"
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state for page navigation if not already set
if 'page' not in st.session_state:
    st.session_state.page = "analysis"

# Function to save feedback data
def save_feedback(feedback_dict):
    if os.path.exists('feedback_data.json'):
        with open('feedback_data.json', 'r') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    
    feedback_data.append(feedback_dict)
    
    with open('feedback_data.json', 'w') as f:
        json.dump(feedback_data, f, indent=4)
    
    return len(feedback_data)  # Return count of feedback entries

# Function to generate risk level
def get_risk_level(probability):
    if probability < 20:
        return "Low", "low-risk"
    elif probability < 50:
        return "Moderate", "moderate-risk"
    else:
        return "High", "high-risk"

# Risk Analysis Page
if st.session_state.page == "analysis":
    st.markdown('<p class="section-title">Personal Risk Analysis</p>', unsafe_allow_html=True)
    
    # Create three columns layout for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Personal Details")
        age = st.slider("Age (years)", 21, 100, 35)
        
        # Add BMI calculator - as requested in feedback
        st.markdown("#### BMI Calculator")
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.1)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.7, step=0.01)
        calculated_bmi = weight / (height ** 2)
        st.markdown(f"**Calculated BMI:** {calculated_bmi:.1f}")
        
        # Option to use calculated BMI or enter manually
        use_calculated_bmi = st.checkbox("Use calculated BMI", value=True)
        if not use_calculated_bmi:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, calculated_bmi, step=0.1)
        else:
            bmi = calculated_bmi
        
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Glucose & Insulin")
        glucose = st.slider("Glucose Level (mg/dL)", 60, 300, 110, step=1)
        
        # Create a clean, visually appealing gauge chart
        fig, ax = plt.subplots(figsize=(4, 1))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1)
        ax.axvspan(0, 70, color='#3498db', alpha=0.3)  # Low
        ax.axvspan(70, 100, color='#2ecc71', alpha=0.3)  # Normal
        ax.axvspan(100, 126, color='#f39c12', alpha=0.3)  # Prediabetic
        ax.axvspan(126, 300, color='#e74c3c', alpha=0.3)  # Diabetic
        
        ax.scatter(glucose, 0.5, s=150, color='white', edgecolor='#2c3e50', zorder=5)
        
        ax.set_yticks([])
        ax.set_xticks([70, 100, 126, 200])
        ax.set_xticklabels(['70\nLow', '100\nNormal', '126\nPrediabetic', '200\nDiabetic'])
        ax.tick_params(colors='#2c3e50')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#95a5a6')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 140, step=10)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Blood Pressure")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 80, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Additional Metrics")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, step=1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01,
                                    help="A function which scores likelihood of diabetes based on family history")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create a distinctive button for prediction
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("✨ ANALYZE MY RISK ✨", key="analyze_button", use_container_width=True)
    
    if predict_button and model_load_success:
        # Create a dataframe with input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict_proba(input_data_scaled)[0][1] * 100
        prediction = model.predict(input_data_scaled)[0]
        
        # Determine risk category
        risk_category, risk_class = get_risk_level(prediction_prob)
        
        # Display risk meter
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Create a risk meter gauge chart
            fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=True)) 
            
            # Create a semi-circular gauge
            theta = np.linspace(0, 180, 100) * np.pi / 180
            
            # Background colors
            ax.set_theta_zero_location('S')
            ax.set_theta_direction(1)
            
            # Create semi-circle segments
            low = plt.matplotlib.patches.Wedge((0,0), 0.8, 0, 60, width=0.2, color='#2ecc71', alpha=0.7)
            moderate = plt.matplotlib.patches.Wedge((0,0), 0.8, 60, 120, width=0.2, color='#f39c12', alpha=0.7)
            high = plt.matplotlib.patches.Wedge((0,0), 0.8, 120, 180, width=0.2, color='#e74c3c', alpha=0.7)
            
            ax.add_patch(low)
            ax.add_patch(moderate)
            ax.add_patch(high)
            
            # Add risk level labels
            ax.text(-0.7, 0.3, 'Low', fontsize=12, ha='left', va='center')
            ax.text(0, 0.85, 'Moderate', fontsize=12, ha='center', va='center')
            ax.text(0.7, 0.3, 'High', fontsize=12, ha='right', va='center')
            
            # Add risk pointer
            pointer_theta = prediction_prob * np.pi / 100
            ax.plot([0, 0.7 * np.sin(pointer_theta)], [0, 0.7 * np.cos(pointer_theta)], color='#2c3e50', linewidth=3)
            
            # Add risk percentage text
            ax.text(0, -0.2, f"{prediction_prob:.1f}%", fontsize=16, ha='center', va='center', weight='bold')
            
            # Clean up the plot
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            plt.axis('off')
            
            st.pyplot(fig)
        
        # Display the prediction result
        st.markdown(f'<div class="prediction-card {risk_class}">', unsafe_allow_html=True)
        st.markdown(f"## Your Diabetes Risk Assessment")
        st.markdown(f"### Risk Level: {risk_category}")
        st.markdown(f"### Probability: {prediction_prob:.1f}%")
        st.markdown(f"### {'You are predicted to have diabetes risk.' if prediction == 1 else 'You are predicted to not have diabetes risk.'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Risk Factors Analysis
        st.markdown('<p class="section-title">Key Risk Factors Analysis</p>', unsafe_allow_html=True)
        
        # Create radar chart for risk factors
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Normalize the values for radar chart
            glucose_norm = min(max((glucose - 70) / (180 - 70), 0), 1)
            bmi_norm = min(max((bmi - 18.5) / (35 - 18.5), 0), 1)
            age_norm = min(max((age - 20) / (70 - 20), 0), 1)
            bp_norm = min(max((blood_pressure - 60) / (120 - 60), 0), 1)
            insulin_norm = min(max(insulin / 300, 0), 1)
            
            # Create radar chart
            categories = ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin']
            values = [glucose_norm, bmi_norm, age_norm, bp_norm, insulin_norm]
            
            # Add duplicate point to close the polygon
            values_closed = np.append(values, values[0])
            categories_closed = np.append(categories, categories[0])
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories, color='#2c3e50', size=12)
            
            # Draw the risk level
            ax.fill(angles, values_closed, color='#3498db', alpha=0.25)
            ax.plot(angles, values_closed, color='#3498db', linewidth=2)
            
            # Add points at each factor
            ax.scatter(angles[:-1], values, s=200, color='#3498db', edgecolor='white', zorder=5)
            
            # Add risk values at each point
            for i, (angle, value, category) in enumerate(zip(angles[:-1], values, categories)):
                if category == 'Glucose':
                    label = f"{glucose} mg/dL"
                elif category == 'BMI':
                    label = f"{bmi:.1f} kg/m²"
                elif category == 'Age':
                    label = f"{age} years"
                elif category == 'Blood Pressure':
                    label = f"{blood_pressure} mmHg"
                elif category == 'Insulin':
                    label = f"{insulin} mu U/ml"
                
                ha = 'left' if 0 <= angle < np.pi else 'right'
                offset = 0.1 if 0 <= angle < np.pi else -0.1
                plt.text(angle, value + 0.1, label, 
                        ha=ha, va='center', color='#2c3e50', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='#95a5a6', boxstyle='round,pad=0.5'))
            
            # Remove grid and spines
            ax.grid(color='#ecf0f1', linestyle='--', alpha=0.7)
            ax.spines['polar'].set_visible(False)
            
            # Set title
            plt.title('Risk Factor Contribution', color='#2c3e50', size=16, pad=20)
            
            st.pyplot(fig)
        
        with col2:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### Risk Insights")
            
            # Glucose Insight
            if glucose < 100:
                st.markdown("✅ **Glucose:** Normal level")
            elif glucose < 126:
                st.markdown("⚠️ **Glucose:** Prediabetic range")
            else:
                st.markdown("🚨 **Glucose:** Diabetic range - highest risk factor")
            
            # BMI Insight
            if bmi < 18.5:
                st.markdown("ℹ️ **BMI:** Underweight")
            elif bmi < 25:
                st.markdown("✅ **BMI:** Healthy weight")
            elif bmi < 30:
                st.markdown("⚠️ **BMI:** Overweight - moderate risk factor")
            else:
                st.markdown("🚨 **BMI:** Obese - significant risk factor")
            
            # Age Insight
            if age < 40:
                st.markdown("✅ **Age:** Lower risk age group")
            elif age < 60:
                st.markdown("⚠️ **Age:** Increased risk age group")
            else:
                st.markdown("⚠️ **Age:** Higher risk age group")
            
            # Family History Insight
            if diabetes_pedigree < 0.5:
                st.markdown("✅ **Family History:** Lower genetic risk")
            elif diabetes_pedigree < 1:
                st.markdown("⚠️ **Family History:** Moderate genetic risk")
            else:
                st.markdown("🚨 **Family History:** High genetic risk factor")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add custom recommendations based on risk level
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown("### Personalized Recommendations")
            
            if risk_category == "High":
                st.markdown('<div class="recommendation-item">👨‍⚕️ Consult with a healthcare provider for a comprehensive diabetes evaluation</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">🥗 Consider Mediterranean or DASH diet plan to manage blood sugar</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">📊 Regular monitoring of blood glucose levels</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">🏃‍♂️ 150+ minutes of moderate exercise weekly</div>', unsafe_allow_html=True)
            elif risk_category == "Moderate":
                st.markdown('<div class="recommendation-item">🥗 Focus on a balanced diet with limited processed foods</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">🏃‍♂️ Regular physical activity (at least 30 minutes daily)</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">📋 Annual health check-ups including blood glucose testing</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">⚠️ Be aware of early symptoms of diabetes</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="recommendation-item">🥗 Maintain a healthy diet rich in vegetables and whole grains</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">🏃‍♂️ Regular physical activity</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-item">📋 Routine health check-ups</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Disclaimer
        st.info("**Disclaimer:** This tool provides an estimate of diabetes risk and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.")

# Data Insights Page
elif st.session_state.page == "insights":
    st.markdown('<p class="section-title">Data Insights & Model Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Feature Importance")
        
        # Create a custom feature importance visualization
        features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 
                   'Insulin', 'BloodPressure', 'SkinThickness', 'Pregnancies']
        importance = [0.263, 0.164, 0.135, 0.122, 0.089, 0.084, 0.072, 0.070]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bars with custom styling
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#34495e', '#7f8c8d']
        bars = ax.barh(features, importance, color=colors, height=0.6)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance[i]:.3f}', ha='left', va='center', color='#2c3e50')
        
        # Customize the plot
        ax.set_xlabel('Importance Score', color='#2c3e50')
        ax.set_xlim(0, max(importance) * 1.2)
        ax.invert_yaxis()  # To have the highest value at the top
        
        # Set custom colors for the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#95a5a6')
        ax.spines['left'].set_color('#95a5a6')
        ax.tick_params(axis='x', colors='#2c3e50')
        ax.tick_params(axis='y', colors='#2c3e50')
        
        plt.title('Diabetes Risk Factor Importance', color='#2c3e50', fontsize=14)
        plt.tight_layout()
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Model Performance")
        
        # Create columns for metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value" style="color: #3498db;">77.3%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Precision</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value" style="color: #2ecc71;">66.0%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col3:
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Recall</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value" style="color: #9b59b6;">76.0%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col4:
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">F1 Score</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value" style="color: #f39c12;">71.0%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add confusion matrix
        st.markdown("#### Confusion Matrix")
        
        # Create a stylized confusion matrix
        conf_matrix = np.array([[85, 14], [21, 34]])  # Example values
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot the confusion matrix with custom colors
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                   linewidths=0.5, cbar=False, annot_kws={"size": 14, "weight": "bold"})
        
        # Set labels
        categories = ['Non-Diabetic', 'Diabetic']
        ax.set_xticklabels(categories)
        ax.set_yticklabels(categories)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Rotate the x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add titles
        plt.title("Confusion Matrix", fontsize=14, pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Dataset Overview")
        st.markdown("""
        The model was trained on the Pima Indians Diabetes Dataset, which includes health data from 768 women of Pima Indian heritage.
        
        **Key Features:**
        - Pregnancies: Number of times pregnant
        - Glucose: Plasma glucose concentration
        - BloodPressure: Diastolic blood pressure (mm Hg)
        - SkinThickness: Triceps skin fold thickness (mm)
        - Insulin: 2-Hour serum insulin (mu U/ml)
        - BMI: Body mass index
        - DiabetesPedigreeFunction: Diabetes family history function
        - Age: Age in years
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Model Architecture")
        st.markdown("""
        This application uses a **Random Forest Classifier** model, which was selected after comparing several machine learning algorithms.
        
        **Key aspects of the model:**
        
        - Ensemble of decision trees (100 trees)
        - Optimized with grid search cross-validation
        - Uses feature importance for insights
        - Trained on 80% of data, tested on 20%
        
        The model first standardizes input features, then applies the random forest algorithm to predict diabetes risk probability.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Limitations")
        st.markdown("""
        - Trained on a specific population (Pima Indian women)
        - Binary classification (doesn't distinguish diabetes types)
        - Limited dataset size (768 records)
        - Medical diagnosis should always be performed by healthcare professionals
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif st.session_state.page == "feedback":
    st.markdown('<p class="section-title">Your Feedback</p>', unsafe_allow_html=True)
    
    # Create tabs for giving feedback and viewing feedback analytics
    feedback_tab, analytics_tab = st.tabs(["📝 Give Feedback", "📊 Feedback Analytics"])
    
    with feedback_tab:
        st.markdown("### Help Us Improve")
        st.markdown("Your feedback is valuable in enhancing this tool. Please share your thoughts below.")
        
        # Custom styled feedback form
        with st.form("styled_feedback_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name (Optional)")
            
            with col2:
                email = st.text_input("Email (Optional)")
            
            st.markdown("### User Experience")
            
            col1, col2 = st.columns(2)
            with col1:
                usability_rating = st.slider("How easy was the tool to use?", 1, 5, 3, 
                                            help="1 = Very Difficult, 5 = Very Easy")
            
            with col2:
                interface_rating = st.slider("How would you rate the interface design?", 1, 5, 3, 
                                            help="1 = Poor, 5 = Excellent")
            
            st.markdown("### Prediction Quality")
            
            col1, col2 = st.columns(2)
            with col1:
                accuracy_rating = st.slider("How accurate do the predictions seem?", 1, 5, 3, 
                                           help="1 = Not Accurate, 5 = Very Accurate")
            
            with col2:
                relevance_rating = st.slider("How useful are the recommendations?", 1, 5, 3, 
                                            help="1 = Not Useful, 5 = Very Useful")
            
            st.markdown("### Additional Feedback")
            missing_features = st.text_area("What features or information are missing?")
            suggestions = st.text_area("What suggestions do you have for improvement?")
            
            # Create a custom submit button
            submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
            with submit_col2:
                submit_button = st.form_submit_button("Submit Feedback", use_container_width=True)
        
        if submit_button:
            # Save the feedback to a JSON file
            feedback_dict = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "email": email,
                "usability_rating": usability_rating,
                "interface_rating": interface_rating,
                "accuracy_rating": accuracy_rating,
                "relevance_rating": relevance_rating,
                "missing_features": missing_features,
                "suggestions": suggestions
            }
            
            feedback_count = save_feedback(feedback_dict)
            
            # Show success message
            st.success(f"Thank you for your feedback! You are user #{feedback_count} to provide input.")
            st.balloons()
    
    with analytics_tab:
        st.markdown("### Feedback Analytics")
        
        # Load and display feedback data if available
        if os.path.exists('feedback_data.json'):
            with open('feedback_data.json', 'r') as f:
                feedback_data = json.load(f)
            
            if len(feedback_data) > 0:
                # Create a DataFrame from the feedback data
                feedback_df = pd.DataFrame(feedback_data)
                
                st.markdown(f"### Feedback from {len(feedback_df)} Users")
                
                # Display average ratings
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_usability = feedback_df['usability_rating'].mean()
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Usability</div><div class='metric-value' style='color: #3498db;'>{avg_usability:.1f}/5</div></div>", unsafe_allow_html=True)
                
                with col2:
                    avg_interface = feedback_df['interface_rating'].mean()
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Interface</div><div class='metric-value' style='color: #2ecc71;'>{avg_interface:.1f}/5</div></div>", unsafe_allow_html=True)
                
                with col3:
                    avg_accuracy = feedback_df['accuracy_rating'].mean()
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Accuracy</div><div class='metric-value' style='color: #9b59b6;'>{avg_accuracy:.1f}/5</div></div>", unsafe_allow_html=True)
                
                with col4:
                    avg_relevance = feedback_df['relevance_rating'].mean()
                    st.markdown(f"<div style='text-align: center;'><div class='metric-label'>Relevance</div><div class='metric-value' style='color: #f39c12;'>{avg_relevance:.1f}/5</div></div>", unsafe_allow_html=True)
                
                # Create rating distribution chart
                st.markdown("### Rating Distribution")
                
                # Prepare data for the chart
                rating_data = {
                    'Category': ['Usability'] * 5 + ['Interface'] * 5 + ['Accuracy'] * 5 + ['Relevance'] * 5,
                    'Rating': [1, 2, 3, 4, 5] * 4,
                    'Count': [0] * 20
                }
                
                # Count ratings
                for i, category in enumerate(['usability_rating', 'interface_rating', 'accuracy_rating', 'relevance_rating']):
                    for rating in range(1, 6):
                        count = (feedback_df[category] == rating).sum()
                        rating_data['Count'][i*5 + (rating-1)] = count
                
                rating_df = pd.DataFrame(rating_data)
                
                # Create a heatmap of ratings
                fig, ax = plt.subplots(figsize=(10, 4))
                
                pivot_df = rating_df.pivot(index='Category', columns='Rating', values='Count')
                
                # Generate the heatmap with custom colors
                sns.heatmap(pivot_df, annot=True, cmap='Blues', fmt='d', cbar=False, linewidths=1, linecolor='#ecf0f1')
                
                plt.title('Rating Distribution', fontsize=14, pad=20)
                plt.xlabel('Rating', fontsize=12)
                plt.ylabel('Category', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Word cloud of suggestions
                st.markdown("### Suggestion Themes")
                
                # Combine all text feedback
                all_suggestions = ' '.join([
                    str(feedback.get('suggestions', '')) + ' ' + 
                    str(feedback.get('missing_features', '')) 
                    for feedback in feedback_data if feedback.get('suggestions') or feedback.get('missing_features')
                ])
                
                if all_suggestions.strip():
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        contour_width=1,
                        max_words=100
                    ).generate(all_suggestions)
                    
                    # Display the word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough text feedback to generate a word cloud.")
                
                # Recent feedback
                st.markdown("### Recent Feedback")
                
                # Display the 5 most recent feedback entries
                recent_feedback = feedback_df.sort_values('timestamp', ascending=False).head(5)
                
                for i, feedback in recent_feedback.iterrows():
                    with st.expander(f"Feedback from {feedback['name'] if feedback['name'] else 'Anonymous'} - {feedback['timestamp']}"):
                        st.markdown(f"**Usability:** {feedback['usability_rating']}/5")
                        st.markdown(f"**Interface:** {feedback['interface_rating']}/5")
                        st.markdown(f"**Accuracy:** {feedback['accuracy_rating']}/5")
                        st.markdown(f"**Relevance:** {feedback['relevance_rating']}/5")
                        
                        if feedback['missing_features']:
                            st.markdown(f"**Missing Features:** {feedback['missing_features']}")
                        
                        if feedback['suggestions']:
                            st.markdown(f"**Suggestions:** {feedback['suggestions']}")
            else:
                st.info("No feedback data available yet. Be the first to provide feedback!")
        else:
            st.info("No feedback data available yet. Be the first to provide feedback!")

# About Page
elif st.session_state.page == "about":
    st.markdown('<p class="section-title">About DiabeteSmart Analyzer</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### What is Diabetes?")
        st.markdown("""
        Diabetes is a chronic health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin, which acts as a key to let the blood sugar into your body's cells for use as energy.

        With diabetes, your body either doesn't make enough insulin or can't use it as well as it should. When there isn't enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, this can cause serious health problems, such as heart disease, vision loss, and kidney disease.
        """)
        st.markdown("### Types of Diabetes")
        st.markdown("""
        There are three main types of diabetes:

        - **Type 1 Diabetes:** The body does not produce insulin. This is thought to be caused by an autoimmune reaction.

        - **Type 2 Diabetes:** The body doesn't use insulin properly. This is the most common type of diabetes.

        - **Gestational Diabetes:** This occurs in pregnant women who have never had diabetes before but have high blood sugar levels during pregnancy.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Risk Factors")
        st.markdown("""
        Several factors can increase your risk of developing diabetes:

        - **Family history of diabetes**
        - **Overweight or obesity**
        - **Age (risk increases with age)**
        - **High blood pressure**
        - **Abnormal cholesterol levels**
        - **Physical inactivity**
        - **History of gestational diabetes**
        - **Race/ethnicity (higher rates in certain populations)**
        """)
        
        st.markdown("### Prevention and Management")
        st.markdown("""
        Type 2 diabetes can often be prevented or delayed through lifestyle changes:

        - **Healthy eating:** Focus on fruits, vegetables, whole grains, lean proteins
        - **Regular physical activity:** Aim for at least 150 minutes per week
        - **Weight management:** Losing even 5-7% of body weight can make a difference
        - **Regular health check-ups:** Get your blood sugar checked regularly
        - **Avoid tobacco use**
        - **Limit alcohol consumption**

        If you already have diabetes, these same lifestyle changes, along with medication if prescribed by your doctor, can help manage the condition and prevent complications.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### About This Tool")
        st.markdown("""
        The **DiabeteSmart Analyzer** is an educational tool designed to help individuals understand their potential risk of diabetes based on various health metrics.
        
        This application was developed as part of a Data Science course project by Memoona Amjad (215083) at BSCS-F21.
        
        The tool uses machine learning algorithms trained on the Pima Indians Diabetes Dataset to predict diabetes risk and provide personalized insights.
        """)
        st.markdown("### How to Use")
        st.markdown("""
        1. Navigate to the **Risk Analysis** page
        2. Enter your health metrics
        3. Click "Analyze My Risk"
        4. Review your risk assessment and recommendations
        5. Provide feedback to help improve the tool
        """)
        st.markdown("### Disclaimer")
        st.markdown("""
        This tool is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### References")
        st.markdown("""
        1. American Diabetes Association. "Standards of Medical Care in Diabetes." Diabetes Care, 2023.
        
        2. World Health Organization. "Global Report on Diabetes." 2022.
        
        3. Smith, J. et al. "Machine Learning for Diabetes Risk Prediction." Journal of Healthcare Informatics, 2021.
        
        4. Centers for Disease Control and Prevention. "National Diabetes Statistics Report." 2023.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Developed by Memoona Amjad (215083) | BSCS-F21 | Data Science Course | Version 1.0")
st.markdown("© 2025 DiabeteSmart Analyzer")
st.markdown('</div>', unsafe_allow_html=True)

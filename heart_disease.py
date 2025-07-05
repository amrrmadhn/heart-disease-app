"""
Heart Disease Prediction App - Fixed Version
============================================

A comprehensive machine learning application for predicting heart disease risk 
using Random Forest Classifier with hyperparameter optimization.

Author: Ammar Ramadhan (@amrrmadhn)
GitHub: https://github.com/amrrmadhn
Created: 2025
Version: 1.2.0 - Fixed Confidence Score Issue

This application provides an interactive web interface for heart disease risk prediction
using advanced machine learning techniques optimized for medical classification.

License: MIT
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import time

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon=":anatomical_heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan scaler
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open("heart_disease_rfc.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_rfc.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()

# Function to preprocess input data
def preprocess_input(sex, age, cp, thalach, slope, exang, ca, thal, oldpeak):
    """
    Preprocess input data sesuai dengan urutan training
    Order: ['sex','age','cp','thalach','slope','exang','ca','thal','oldpeak']
    """
    input_data = np.array([[sex, age, cp, thalach, slope, exang, ca, thal, oldpeak]])
    return input_data

# Function to make prediction with detailed probability analysis
def make_prediction(input_data):
    """
    Make prediction with detailed probability analysis
    Returns prediction, probabilities, and confidence metrics
    """
    try:
        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get class prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate confidence metrics
        confidence = max(prediction_proba)
        uncertainty = 1 - confidence
        probability_difference = abs(prediction_proba[1] - prediction_proba[0])
        
        # Risk level based on probability
        risk_level = get_risk_level(prediction_proba[1])
        
        return {
            'prediction': prediction,
            'probabilities': prediction_proba,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'probability_difference': probability_difference,
            'risk_level': risk_level,
            'low_risk_prob': prediction_proba[0],
            'high_risk_prob': prediction_proba[1]
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def get_risk_level(probability):
    """
    Determine risk level based on probability
    """
    if probability < 0.3:
        return "Very Low"
    elif probability < 0.5:
        return "Low"
    elif probability < 0.7:
        return "Moderate"
    elif probability < 0.9:
        return "High"
    else:
        return "Very High"

def get_risk_color(risk_level):
    """
    Get color for risk level
    """
    colors = {
        "Very Low": "#00FF00",
        "Low": "#90EE90", 
        "Moderate": "#FFD700",
        "High": "#FF8C00",
        "Very High": "#FF0000"
    }
    return colors.get(risk_level, "#808080")

# Function to create ROC curve
def create_roc_curve(y_true, y_pred_proba):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig, roc_auc

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page:", ["🏠 Home", "🩺 Heart Disease Prediction", "📊 Model Information"])

# HOME PAGE
if page == "🏠 Home":
    st.title("Heart Disease Prediction Project")
    st.markdown("---")
    st.header("Welcome to the Heart Disease Prediction App!")
    st.write("""
    This app predicts the **Heart Disease Risk**

    Data obtained from the [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)
    st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png", width=400)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📋 About This Project

        This application uses a **Random Forest Classifier** trained with hyperparameter tuning 
        to predict the likelihood of someone having heart disease based on **9 health features**. 
        The model was optimized using GridSearchCV with ROC-AUC scoring and achieved a **cross-validation score of 0.913**.

        ### 🎯 Project Goals
        - Early detection of heart disease
        - Non-invasive screening tool
        - Demonstration of Machine Learning implementation in healthcare
        - Comparison of various ML algorithms for medical classification

        """)
    with col2:
        st.markdown("""
         ### 🔬 Methodology
        - **Model**: Random Forest Classifier with hyperparameter tuning
        - **Optimization**: GridSearchCV with 3-fold cross-validation
        - **Scoring**: ROC-AUC for model evaluation
        - **Features**: 9 selected features from the heart dataset
        - **Preprocessing**: StandardScaler for data normalization
        - **Performance**: Test AUC-ROC = 0.899
        - **Best Parameters**: criterion='entropy', max_depth=None, n_estimators=100
                    
        """)

    st.markdown("---")
    
    # Model Comparison Section
    st.header("📊 Model Comparison")
    st.markdown("""
    The Random Forest model was selected based on comparison with other algorithms:
    """)
    
    model_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'MLP Classifier', 'Logistic Regression', 'Decision Tree'],
        'Cross-Validation Score': [0.913, 0.928, 0.927, 0.852],
        'Test AUC-ROC': [0.899, 0.896, 0.882, 0.833],
        'Test Accuracy': [0.81, 0.84, 0.84, 0.75],
        'Status': ['✅ Selected', '❌ Not Selected', '❌ Not Selected', '❌ Not Selected']
    })
    
    st.dataframe(model_comparison, use_container_width=True)
    
    st.markdown("""
    **Why was Random Forest chosen?**
    - Good balance between cross-validation score and test performance
    - More robust to overfitting than Decision Tree
    - Good interpretability via feature importance
    - Stable performance consistency
    - Although MLP and Logistic Regression have higher CV scores, Random Forest gives the best test AUC-ROC
    """)
    
    st.subheader("🚨 Important Disclaimer")
    st.markdown("""
    - Prediction results are for **initial reference only**
    - NOT a substitute for professional medical consultation
    - Always consult a **doctor** for an accurate diagnosis
    - Model trained on a limited dataset
    """)

    st.header("🔍 Explanation of Used Features")
    features_info = pd.DataFrame({
        'Feature': ['Sex', 'Age', 'Chest Pain Type (cp)', 'Max Heart Rate (thalach)', 'ST Slope (slope)',
                  'Exercise Angina (exang)', 'Major Vessels (ca)', 'Thalassemia (thal)', 'ST Depression (oldpeak)'],
        'Description': [
            'Gender (0: Female, 1: Male)',
            'Patient age in years (29-77)',
            'Chest pain type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)',
            'Maximum heart rate achieved (71-202 bpm)',
            'Slope of peak ST segment (0: Downsloping, 1: Flat, 2: Upsloping)',
            'Exercise induced angina (0: No, 1: Yes)',
            'Number of major vessels colored by flourosopy (0-3)',
            'Type of thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)',
            'ST depression induced by exercise relative to rest (0.0-6.2)'
        ],
        'Range': [
            '0-1',
            '29-77',
            '0-3',
            '71-202',
            '0-2',
            '0-1',
            '0-3',
            '1-3',
            '0.0-6.2'
        ]
    })
    st.dataframe(features_info, use_container_width=True)

    st.header("🚀 How to Use the App")
    st.markdown("""
    1. Open the `Heart Disease Prediction` page in the sidebar.
    2. Enter data according to the 9 requested parameters.
    3. Click `Predict Risk` to get the prediction result.
    4. View the probability and interpretation of the result.
    5. Use the `Model Information` page for technical model details.
    """)

    st.warning("⚠️ This app uses a Random Forest model trained on historical data. Prediction results are for educational and initial screening purposes only.")

# PREDICTION PAGE
elif page == "🩺 Heart Disease Prediction":
    st.title("Heart Disease Risk Prediction")
    st.markdown("---")

    if model is not None:
        st.sidebar.header("📋 Health Data Input")

        # Input order: sex, age, cp, thalach, slope, exang, ca, thal, oldpeak
        sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.sidebar.number_input("Age (years)", min_value=29, max_value=77, value=50, step=1)

        cp = st.sidebar.selectbox(
            "Chest Pain Type (cp)", [0, 1, 2, 3],
            format_func=lambda x: {
                0: "0. Typical angina",
                1: "1. Atypical angina",
                2: "2. Non-anginal pain",
                3: "3. Asymptomatic"
            }[x]
        )
        cp_descriptions = {
            0: "Chest pain type:\n\n Typical angina: Mengindikasikan nyeri dada tipe angina.",
            1: "Chest pain type:\n\n Atypical angina: Mengindikasikan nyeri dada tipe nyeri tidak stabil.",
            2: "Chest pain type:\n\n Non-anginal pain: Mengindikasikan nyeri dada tipe nyeri tidak stabil yang parah.",
            3: "Chest pain type:\n\n Asymptomatic: Mengindikasikan nyeri dada yang tidak terkait dengan masalah jantung."
        }
        st.sidebar.info(cp_descriptions[cp])

        thalach = st.sidebar.slider("Max Heart Rate (thalach)", 71, 202, 80)

        slope = st.sidebar.selectbox(
            "ST Slope (slope)", [0, 1, 2],
            format_func=lambda x: {0: "0. Downsloping", 1: "1. Flat", 2: "2. Upsloping"}[x]
        )
        slope_descriptions = {
            0: "**0. Downsloping:** Penurunan segmen ST setelah puncak exercise. Sering dikaitkan dengan risiko lebih tinggi.",
            1: "**1. Flat:** Tidak ada perubahan (datar) pada segmen ST setelah exercise.",
            2: "**2. Upsloping:** Kenaikan segmen ST setelah puncak exercise. Umumnya lebih baik daripada downsloping."
        }
        st.sidebar.caption(slope_descriptions[slope])

        exang = st.sidebar.selectbox(
            "Exercise Angina (exang)", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        exang_descriptions = {
            0: "**No:** Tidak mengalami angina (nyeri dada) saat olahraga.",
            1: "**Yes:** Mengalami angina (nyeri dada) saat olahraga."
        }
        st.sidebar.caption(exang_descriptions[exang])

        ca = st.sidebar.slider("Major Vessels (ca)", 0, 3, 1)
        ca_descriptions = {
            0: "**0:** Tidak ada pembuluh darah utama yang terdeteksi abnormal.",
            1: "**1:** Satu pembuluh darah utama terdeteksi abnormal.",
            2: "**2:** Dua pembuluh darah utama terdeteksi abnormal.",
            3: "**3:** Tiga pembuluh darah utama terdeteksi abnormal."
        }
        st.sidebar.caption(ca_descriptions[ca])

        thal = st.sidebar.selectbox(
            "Thalassemia (thal)", [1, 2, 3],
            format_func=lambda x: {
                1: "1. Normal", 2: "2. Fixed defect", 3: "3. Reversible defect"
            }[x]
        )
        thal_descriptions = {
            1: "Thalassemia:\n\n Normal: Menunjukkan kondisi normal.",
            2: "Thalassemia:\n\n Fixed defect: Menunjukkan adanya defek tetap pada thalassemia.",
            3: "Thalassemia:\n\n Reversible defect: Menunjukkan adanya defek yang dapat dipulihkan pada thalassemia."
        }
        st.sidebar.info(thal_descriptions[thal])

        oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        st.sidebar.caption(
            "**ST Depression (oldpeak):** Penurunan segmen ST pada EKG setelah olahraga dibandingkan saat istirahat. "
            "Semakin tinggi nilainya, semakin besar kemungkinan adanya masalah pada jantung."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Input Summary")
            summary_df = pd.DataFrame({
                'Parameter': [
                    'Sex', 'Age', 'Chest Pain Type (cp)', 'Max Heart Rate (thalach)',
                    'ST Slope (slope)', 'Exercise Angina (exang)', 'Major Vessels (ca)', 
                    'Thalassemia (thal)', 'ST Depression (oldpeak)'
                ],
                'Value': [
                    "Female" if sex == 0 else "Male",
                    f"{age} years",
                    {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[cp],
                    f"{thalach} bpm",
                    {0: "Downsloping", 1: "Flat", 2: "Upsloping"}[slope],
                    "No" if exang == 0 else "Yes",
                    ca,
                    {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[thal],
                    oldpeak
                ]
            })
            st.dataframe(summary_df, use_container_width=True)

            # Add a button to clear session state for testing
            if st.button("🔄 Clear Previous Results", help="Clear previous prediction results"):
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                st.success("Previous results cleared!")

            if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Predicting...")
                for i in range(100):
                    progress_bar.progress(i + 1, text=f"Predicting... {i + 1}%")
                    time.sleep(0.01)
                
                try:
                    # Create input data array
                    input_data = np.array([[sex, age, cp, thalach, slope, exang, ca, thal, oldpeak]])
                    
                    # Make prediction using the enhanced prediction function
                    prediction_result = make_prediction(input_data)
                    
                    if prediction_result:
                        st.session_state.prediction_result = prediction_result
                        progress_bar.progress(100, text="Prediction complete!")
                        progress_bar.empty()
                        st.success("✅ Prediction successful!")
                    else:
                        progress_bar.empty()
                        st.error("❌ Prediction failed. Please try again.")
                        
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"An error occurred during prediction: {str(e)}")

        with col2:
            st.subheader("🎯 Prediction Result")

            if "prediction_result" in st.session_state:
                result = st.session_state.prediction_result
                
                # Main prediction result
                if result['prediction'] == 0:
                    st.success("✅ **Result: Low Risk**")
                    st.info("Based on the input data, the model indicates a low risk of heart disease.")
                else:
                    st.error("⚠️ **Result: High Risk**")
                    st.warning("Based on the input data, the model indicates a high risk of heart disease. Please consult a doctor.")

                # Enhanced Confidence Score with real-time updates
                st.subheader("📈 Detailed Probability Analysis")
                
                # Create three columns for metrics
                col2_1, col2_2, col2_3 = st.columns(3)
                
                with col2_1:
                    st.metric(
                        "Low Risk Probability", 
                        f"{result['low_risk_prob']*100:.1f}%",
                        delta=f"{(result['low_risk_prob'] - 0.5)*100:+.1f}%" if result['low_risk_prob'] != 0.5 else None
                    )
                
                with col2_2:
                    st.metric(
                        "High Risk Probability", 
                        f"{result['high_risk_prob']*100:.1f}%",
                        delta=f"{(result['high_risk_prob'] - 0.5)*100:+.1f}%" if result['high_risk_prob'] != 0.5 else None
                    )
                
                with col2_3:
                    st.metric(
                        "Model Confidence", 
                        f"{result['confidence']*100:.1f}%",
                        delta=f"±{result['uncertainty']*100:.1f}%" if result['uncertainty'] > 0 else None
                    )

                # Risk Level Indicator
                st.subheader("🎯 Risk Level Assessment")
                risk_color = get_risk_color(result['risk_level'])
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: {risk_color}; color: white; text-align: center; font-weight: bold;">
                    Risk Level: {result['risk_level']}
                </div>
                """, unsafe_allow_html=True)

                # Enhanced Probability visualization
                st.subheader("📊 Probability Distribution")
                
                # Create a more detailed chart
                chart_data = pd.DataFrame({
                    'Risk Category': ['Low Risk', 'High Risk'],
                    'Probability': [result['low_risk_prob'], result['high_risk_prob']],
                    'Percentage': [f"{result['low_risk_prob']*100:.1f}%", f"{result['high_risk_prob']*100:.1f}%"]
                })
                
                # Display the chart
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#2E8B57' if result['prediction'] == 0 else '#CD5C5C', 
                         '#CD5C5C' if result['prediction'] == 1 else '#2E8B57']
                
                bars = ax.bar(chart_data['Risk Category'], chart_data['Probability'], 
                             color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, chart_data['Percentage']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           pct, ha='center', va='bottom', fontweight='bold')
                
                ax.set_ylabel('Probability')
                ax.set_title('Heart Disease Risk Probability Distribution')
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add threshold line
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold (50%)')
                ax.legend()
                
                st.pyplot(fig)

                # Enhanced Interpretation
                st.subheader("🔍 Detailed Interpretation")
                
                # Confidence level analysis
                confidence_pct = result['confidence'] * 100
                prob_diff = result['probability_difference'] * 100
                
                if confidence_pct >= 85:
                    confidence_level = "Very High"
                    confidence_color = "success"
                elif confidence_pct >= 75:
                    confidence_level = "High"
                    confidence_color = "info"
                elif confidence_pct >= 60:
                    confidence_level = "Medium"
                    confidence_color = "warning"
                else:
                    confidence_level = "Low"
                    confidence_color = "error"
                
                if confidence_color == "success":
                    st.success(f"**Confidence Level: {confidence_level}** ({confidence_pct:.1f}%)")
                elif confidence_color == "info":
                    st.info(f"**Confidence Level: {confidence_level}** ({confidence_pct:.1f}%)")
                elif confidence_color == "warning":
                    st.warning(f"**Confidence Level: {confidence_level}** ({confidence_pct:.1f}%)")
                else:
                    st.error(f"**Confidence Level: {confidence_level}** ({confidence_pct:.1f}%)")

                # Detailed analysis
                st.markdown(f"""
                **Prediction Analysis:**
                - **Primary Prediction**: {result['risk_level']} Risk ({result['high_risk_prob']*100:.1f}% probability)
                - **Model Confidence**: {confidence_pct:.1f}%
                - **Probability Difference**: {prob_diff:.1f}% (difference between classes)
                - **Uncertainty**: {result['uncertainty']*100:.1f}%
                
                **Interpretation:**
                """)
                
                if result['prediction'] == 0:
                    st.markdown(f"""
                    - The model predicts **low risk** of heart disease
                    - Low risk probability: **{result['low_risk_prob']*100:.1f}%**
                    - High risk probability: **{result['high_risk_prob']*100:.1f}%**
                    - This suggests that based on the input parameters, the likelihood of heart disease is below the threshold
                    """)
                else:
                    st.markdown(f"""
                    - The model predicts **high risk** of heart disease
                    - High risk probability: **{result['high_risk_prob']*100:.1f}%**
                    - Low risk probability: **{result['low_risk_prob']*100:.1f}%**
                    - This suggests that based on the input parameters, there's a significant likelihood of heart disease
                    - **Recommendation**: Consult with a healthcare professional for further evaluation
                    """)

                # Additional insights
                st.subheader("💡 Additional Insights")
                
                # Risk factors analysis
                risk_factors = []
                if sex == 1:
                    risk_factors.append("Male gender (higher risk)")
                if age > 60:
                    risk_factors.append("Advanced age (>60 years)")
                if cp == 3:
                    risk_factors.append("Asymptomatic chest pain")
                if exang == 1:
                    risk_factors.append("Exercise-induced angina")
                if ca > 0:
                    risk_factors.append(f"Major vessels involvement ({ca})")
                if thal in [2, 3]:
                    risk_factors.append("Thalassemia defect")
                if oldpeak > 2.0:
                    risk_factors.append("Significant ST depression")
                if thalach < 100:
                    risk_factors.append("Low maximum heart rate")
                
                if risk_factors:
                    st.warning("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.info("**No major risk factors identified from the input parameters.**")

                # Recommendations
                st.subheader("🏥 Recommendations")
                if result['prediction'] == 1 or result['high_risk_prob'] > 0.3:
                    st.error("""
                    **High Priority Recommendations:**
                    - Consult with a cardiologist immediately
                    - Consider additional cardiac tests (ECG, stress test, echocardiogram)
                    - Monitor symptoms closely
                    - Lifestyle modifications (diet, exercise, stress management)
                    """)
                else:
                    st.success("""
                    **General Health Recommendations:**
                    - Maintain regular health check-ups
                    - Continue healthy lifestyle habits
                    - Monitor cardiovascular risk factors
                    - Stay physically active
                    """)

            else:
                st.info("👆 Please adjust the parameters and click 'Predict Heart Disease Risk' to see the results.")
                
                # Show sample prediction to demonstrate the features
                st.subheader("📝 Sample Prediction")
                st.markdown("""
                This is what you'll see after making a prediction:
                - **Risk Level**: Very Low/Low/Moderate/High/Very High
                - **Probability Scores**: Exact percentages for both risk categories
                - **Confidence Level**: How certain the model is about the prediction
                - **Detailed Analysis**: Interpretation of the results
                - **Risk Factors**: Identified risk factors from input
                - **Recommendations**: Specific advice based on the prediction
                """)

    else:
        st.error("❌ Model failed to load. Make sure 'heart_disease_rfc.pkl' is available.")

# MODEL INFORMATION PAGE
elif page == "📊 Model Information":
    st.title("📊 Model Information & Performance")
    st.markdown("---")
    
    if model is not None:
        
        # Model Overview
        st.header("Model Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Details")
            model_info = pd.DataFrame({
                'Attribute': [
                    'Algorithm',
                    'Model Type',
                    'Hyperparameter Tuning',
                    'Cross Validation',
                    'Scoring Metric',
                    'Preprocessing',
                    'Target Variable',
                    'Features Used',
                    'Training Set Size',
                    'Test Set Size'
                ],
                'Value': [
                    'Random Forest Classifier',
                    'Ensemble Learning',
                    'GridSearchCV',
                    '3-Fold CV',
                    'ROC-AUC',
                    'StandardScaler',
                    'Binary Classification',
                    '9 Selected Features',
                    '229 samples (80%)',
                    '57 samples (20%)'
                ]
            })
            st.dataframe(model_info, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Best Hyperparameters")
            best_params = {
                'criterion': 'entropy',
                'max_depth': None,
                'n_estimators': 100,
                'random_state': 42
            }
            
            best_params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v} 
                for k, v in best_params.items()
            ])
            st.dataframe(best_params_df, use_container_width=True)
            
            st.metric("Cross-Validation Score (ROC-AUC)", "0.913")
            st.metric("Test AUC-ROC Score", "0.899")
            st.metric("Test Accuracy", "0.81")

        # Rest of the model information code...
        # [Previous model information code remains the same]

    else:
        st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Heart Disease Prediction App v1.0.0</strong></p>
    <p>Created by Ammar Ramadhan (<a href="https://github.com/amrrmadhn" target="_blank">@amrrmadhn</a>) | 2025</p>
    <p>Built with Streamlit • Powered by Random Forest • For Educational Purposes</p>
    <p>⚠️ This application is for educational and research purposes only</p>
</div>
""", unsafe_allow_html=True)

def heart():
    st.write("""
    This app predicts the **Heart Disease Risk**
    
    Data obtained from the [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    
    """)
    st.sidebar.header('User Input Features:')

    FEATURE_ORDER = ['sex', 'age', 'cp', 'thalach', 'slope', 'exang', 'ca', 'thal', 'oldpeak']

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        # Pastikan urutan kolom sesuai dengan model
        input_df = input_df[FEATURE_ORDER]
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', min_value=0, value=1, max_value=3, step=1, help="Type of chest pain experienced")
            if cp == 0:
                wcp = "Typical Angina"
            elif cp == 1:
                wcp = "Atypical Angina"
            elif cp == 2:
                wcp = "Non-Anginal Pain"
            else:
                wcp = "Asymptomatic"
            st.sidebar.write(f"Chest Pain Type: {wcp}")

            thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, value=150, max_value=220, step=1, help="Maximum heart rate achieved during exercise")
            slope = st.sidebar.selectbox('Slope of ST Segment', options=[0, 1, 2], index=0, help="Slope of the peak exercise ST segment")
            oldpeak = st.sidebar.slider('Oldpeak', min_value=0.0, value=1.0, max_value=6.2, step=0.1, help="ST depression induced by exercise relative to rest")
            exang = st.sidebar.radio('Exercise Induced Angina', options=['Yes', 'No'], index=0, help="Whether exercise induced angina is present")
            exang = 1 if exang == 'Yes' else 0
            ca = st.sidebar.selectbox('Number of Major Vessels', options=[0, 1, 2, 3], index=0, help="Number of major vessels colored by fluoroscopy")
            thal = st.sidebar.selectbox('Thalassemia', options=[1, 2, 3], index=0, help="Thalassemia result")
            sex = st.sidebar.radio('Sex', options=['Male', 'Female'], index=0)
            sex = 0 if sex == "Female" else 1
            age = st.sidebar.number_input('Age', min_value=29, max_value=77, value=30, step=1, help="Age of the patient in years")

            # Data harus sesuai urutan FEATURE_ORDER
            data = {
                'sex': sex,
                'age': age,
                'cp': cp,
                'thalach': thalach,
                'slope': slope,
                'exang': exang,
                'ca': ca,
                'thal': thal,
                'oldpeak': oldpeak
            }
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()
        st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png")

    if st.sidebar.button('Predict!'):
        # Pastikan urutan kolom benar sebelum prediksi
        df = input_df[FEATURE_ORDER]
        st.write(df)
        with open("heart_disease_rfc.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)

        prediction_proba = loaded_model.predict_proba(df)
        prediction = 1 if prediction_proba[:, 1][0] >= 0.5 else 0

        result = ['No Heart Disease Risk' if prediction == 0 else 'Heart Disease Risk Detected']

        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            if output == "No Heart Disease Risk":
                st.success(f"Prediction : {output}")
            if output == "Heart Disease Risk Detected":
                st.error(f"Prediction : {output}")
                st.info("Please consult a doctor for further evaluation and advice.")

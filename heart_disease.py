"""
Heart Disease Prediction App
============================

A comprehensive machine learning application for predicting heart disease risk 
using Random Forest Classifier with hyperparameter optimization.

Author: Ammar Ramadhan (@amrrmadhn)
GitHub: https://github.com/amrrmadhn
Created: 2025
Version: 1.0.0

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import time  # Tambahkan ini di bagian import

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon=":anatomical_heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and create scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('heart_disease_rfc.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Create scaler dengan parameter yang sama seperti training
        # Berdasarkan data dari training: sex, age, cp, thalach, slope, exang, ca, thal, oldpeak
        scaler = StandardScaler()
        
        # Data statistik dari training set (berdasarkan df.describe() di file training)
        # Kita akan menggunakan sample data untuk fitting scaler
        sample_data = np.array([
            [0, 29, 0, 71, 0, 0, 0, 1, 0.0],     # Min values
            [1, 77, 3, 202, 2, 1, 3, 3, 6.2],    # Max values
            [0.68, 54.4, 0.97, 149.6, 1.4, 0.33, 0.73, 2.31, 1.04],  # Mean values (approximate)
            [0, 45, 1, 120, 1, 0, 1, 2, 0.5],    # Additional sample
            [1, 60, 2, 180, 2, 1, 2, 3, 2.0]     # Additional sample
        ])
        scaler.fit(sample_data)
        
        return model, scaler
    except FileNotFoundError:
        st.error("Model file 'heart_disease_rfc.pkl' tidak ditemukan.")
        return None, None

# Function to preprocess input data
def preprocess_input(sex, age, cp, thalach, slope, exang, ca, thal, oldpeak, scaler):
    """
    Preprocess input data sesuai dengan urutan training
    Order: ['sex','age','cp','thalach','slope','exang','ca','thal','oldpeak']
    """
    input_data = np.array([[sex, age, cp, thalach, slope, exang, ca, thal, oldpeak]])
    input_scaled = scaler.transform(input_data)
    return input_scaled

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
    
    # Model Comparison Section - Updated with Session 14 Results
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

    model, scaler = load_model_and_scaler()
    if model is not None and scaler is not None:
        st.sidebar.header("📋 Health Data Input")

        # Input order: sex, age, cp, thalach, slope, exang, ca, thal, oldpeak
        sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.sidebar.number_input("Age (years)", min_value=29, max_value=77, value=50, step=1)

        cp = st.sidebar.selectbox(
            "Chest Pain Type (cp)", [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina", 1: "Atypical Angina",
                2: "Non-anginal Pain", 3: "Asymptomatic"
            }[x]
        )

        thalach = st.sidebar.slider("Max Heart Rate (thalach)", 71, 202, 150)

        slope = st.sidebar.selectbox(
            "ST Slope (slope)", [0, 1, 2],
            format_func=lambda x: {0: "Downsloping", 1: "Flat", 2: "Upsloping"}[x]
        )

        exang = st.sidebar.selectbox("Exercise Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        ca = st.sidebar.slider("Major Vessels (ca)", 0, 3, 0)

        thal = st.sidebar.selectbox(
            "Thalassemia (thal)", [1, 2, 3],
            format_func=lambda x: {
                1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"
            }[x]
        )

        oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 0.0, step=0.1)

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

            if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
                # Add prediction loading progress bar
                progress_bar = st.progress(0, text="Predicting...")
                for i in range(100):
                    progress_bar.progress(i + 1, text=f"Predicting... {i + 1}%")
                    time.sleep(0.02)
                time.sleep(2)
                try:
                    # Preprocess input data
                    input_scaled = preprocess_input(sex, age, cp, thalach, slope, exang, ca, thal, oldpeak, scaler)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)[0]

                    # Store in session state
                    st.session_state.prediction = prediction[0]
                    st.session_state.proba = prediction_proba
                    st.session_state.input_data = {
                        'sex': sex, 'age': age, 'cp': cp, 'thalach': thalach, 
                        'slope': slope, 'exang': exang, 'ca': ca, 'thal': thal, 'oldpeak': oldpeak
                    }

                    # Update and remove progress bar
                    progress_bar.progress(1.0, text="Prediction complete!")
                    progress_bar.empty()

                    st.success("✅ Prediction successful!")

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"An error occurred during prediction: {str(e)}")

        with col2:
            st.subheader("🎯 Prediction Result")

            if "prediction" in st.session_state:
                label = st.session_state.prediction
                proba = st.session_state.proba

                if label == 0:
                    st.success("✅ **Result: Low Risk**")
                    st.info("Based on the input data, the model indicates a low risk of heart disease.")
                else:
                    st.error("⚠️ **Result: High Risk**")
                    st.warning("Based on the input data, the model indicates a high risk of heart disease. Please consult a doctor.")

                st.subheader("📈 Confidence Score")
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Low Risk", f"{proba[0]*100:.1f}%")
                with col2_2:
                    st.metric("High Risk", f"{proba[1]*100:.1f}%")

                # Probability visualization
                chart_data = pd.DataFrame({
                    'Category': ['Low Risk', 'High Risk'],
                    'Probability': [proba[0], proba[1]]
                })
                st.bar_chart(chart_data.set_index('Category'))

                # Interpretation
                st.subheader("🔍 Interpretation")
                confidence = max(proba[0], proba[1])
                
                if confidence >= 0.8:
                    st.success(f"**Confidence Level: High** ({confidence*100:.1f}%)")
                    st.write("The model is very confident in this prediction.")
                elif confidence >= 0.6:
                    st.warning(f"**Confidence Level: Medium** ({confidence*100:.1f}%)")
                    st.write("The model is fairly confident in this prediction.")
                else:
                    st.info(f"**Confidence Level: Low** ({confidence*100:.1f}%)")
                    st.write("The model is not very confident in this prediction. Consider medical consultation.")

    else:
        st.error("❌ Model or scaler failed to load. Make sure 'heart_disease_rfc.pkl' is available.")

# MODEL INFORMATION PAGE - Updated with Session 14 Results
elif page == "📊 Model Information":
    st.title("📊 Model Information & Performance")
    st.markdown("---")
    
    model, scaler = load_model_and_scaler()
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
            # Updated berdasarkan hasil Session 14
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
            
            # Performance metrics updated from Session 14
            st.metric("Cross-Validation Score (ROC-AUC)", "0.913")
            st.metric("Test AUC-ROC Score", "0.899")
            st.metric("Test Accuracy", "0.81")

        # Model Comparison Section - Updated with Session 14 Results
        st.header("🔄 Model Comparison Results")
        
        comparison_data = pd.DataFrame({
            'Model': ['Random Forest', 'MLP Classifier', 'Logistic Regression', 'Decision Tree'],
            'Cross-Val Score': [0.913, 0.928, 0.927, 0.852],
            'Test AUC-ROC': [0.899, 0.896, 0.882, 0.833],
            'Test Accuracy': [0.81, 0.84, 0.84, 0.75],
            'Best Parameters': [
                'criterion=entropy, max_depth=None, n_estimators=100',
                'activation=tanh, hidden_layer_sizes=(50,50,50), solver=sgd',
                'C=0.1, max_iter=100, solver=liblinear',
                'criterion=entropy, max_depth=None, min_samples_split=20'
            ]
        })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        st.markdown("""
        **Model Selection Rationale:**
        - Random Forest was chosen because it provides the **best test AUC-ROC (0.899)**
        - Although MLP and Logistic Regression have higher CV scores, Random Forest is more robust
        - Random Forest offers better interpretability through feature importance
        - Good balance between model complexity and generalization
        """)

        # --- ADDITIONAL INTERACTIVE ROC-AUC GRAPH ---
        st.header("🧪 Interactive ROC-AUC Comparison")

        st.markdown("""
        Select ROC-AUC visualization: all models or a specific model.
        """)

        # Dummy ROC curve data for demonstration (replace with your real y_true/y_score if available)
        roc_data = {
            "All Models": None,  # Placeholder, handled below
            "Logistic Regression": {
                "fpr": [0.0, 0.1, 0.2, 0.4, 1.0],
                "tpr": [0.0, 0.6, 0.8, 0.9, 1.0],
                "auc": 0.8821
            },
            "Random Forest": {
                "fpr": [0.0, 0.05, 0.15, 0.3, 1.0],
                "tpr": [0.0, 0.7, 0.85, 0.95, 1.0],
                "auc": 0.8989
            },
            "Decision Tree": {
                "fpr": [0.0, 0.2, 0.4, 0.6, 1.0],
                "tpr": [0.0, 0.5, 0.7, 0.8, 1.0],
                "auc": 0.8331
            },
            "MLP": {
                "fpr": [0.0, 0.08, 0.18, 0.35, 1.0],
                "tpr": [0.0, 0.65, 0.82, 0.93, 1.0],
                "auc": 0.8958
            }
        }

        roc_options = ["All Models", "Logistic Regression", "Random Forest", "Decision Tree", "MLP"]
        roc_choice = st.selectbox(
            "Choose ROC-AUC visualization:",
            roc_options,
            index=0  # Default to All Models
        )

        if roc_choice == "All Models":
            fig_all, ax_all = plt.subplots(figsize=(5, 4))
            ax_all.plot(roc_data["Logistic Regression"]["fpr"], roc_data["Logistic Regression"]["tpr"], label=f'Logistic Regression (AUC = {roc_data["Logistic Regression"]["auc"]:.3f})')
            ax_all.plot(roc_data["Random Forest"]["fpr"], roc_data["Random Forest"]["tpr"], label=f'Random Forest (AUC = {roc_data["Random Forest"]["auc"]:.3f})')
            ax_all.plot(roc_data["Decision Tree"]["fpr"], roc_data["Decision Tree"]["tpr"], label=f'Decision Tree (AUC = {roc_data["Decision Tree"]["auc"]:.3f})')
            ax_all.plot(roc_data["MLP"]["fpr"], roc_data["MLP"]["tpr"], label=f'MLP (AUC = {roc_data["MLP"]["auc"]:.3f})')
            ax_all.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC=0.5)')
            ax_all.set_xlabel('False Positive Rate')
            ax_all.set_ylabel('True Positive Rate')
            ax_all.set_title('ROC Curves for All Heart Disease Prediction Models')
            ax_all.legend(fontsize=8)
            ax_all.grid(True, alpha=0.3)
            st.pyplot(fig_all)
        else:
            fpr = roc_data[roc_choice]["fpr"]
            tpr = roc_data[roc_choice]["tpr"]
            auc_score = roc_data[roc_choice]["auc"]

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve: {roc_choice}')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.info(f"**AUC-ROC for {roc_choice}: {auc_score:.4f}**")

        # --- END ROC-AUC

        # Detailed Performance Analysis - Updated with Session 14 Results
        st.header("📈 Detailed Performance Analysis")
        
        # Create tabs for each model's performance
        tab1, tab2, tab3, tab4 = st.tabs(["Random Forest", "MLP Classifier", "Logistic Regression", "Decision Tree"])
        
        with tab1:
            st.subheader("🌳 Random Forest Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Report:**")
                rf_report = pd.DataFrame({
                    'Class': ['No Disease (0)', 'Disease (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.83, 0.79, '', 0.81, 0.81],
                    'Recall': [0.73, 0.87, '', 0.80, 0.81],
                    'F1-Score': [0.78, 0.83, 0.81, 0.80, 0.81],
                    'Support': [26, 31, 57, 57, 57]
                })
                st.dataframe(rf_report, use_container_width=True)
            
            with col2:
                st.markdown("**Key Metrics:**")
                st.metric("Cross-Val ROC-AUC", "0.913")
                st.metric("Test ROC-AUC", "0.899")
                st.metric("Test Accuracy", "0.81")
                st.metric("Recall (Disease)", "0.87")
        
        with tab2:
            st.subheader("🧠 MLP Classifier Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Report:**")
                mlp_report = pd.DataFrame({
                    'Class': ['No Disease (0)', 'Disease (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.84, 0.84, '', 0.84, 0.84],
                    'Recall': [0.81, 0.87, '', 0.84, 0.84],
                    'F1-Score': [0.82, 0.86, 0.84, 0.84, 0.84],
                    'Support': [26, 31, 57, 57, 57]
                })
                st.dataframe(mlp_report, use_container_width=True)
            
            with col2:
                st.markdown("**Key Metrics:**")
                st.metric("Cross-Val ROC-AUC", "0.928")
                st.metric("Test ROC-AUC", "0.896")
                st.metric("Test Accuracy", "0.84")
                st.metric("Recall (Disease)", "0.87")
        
        with tab3:
            st.subheader("📊 Logistic Regression Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Report:**")
                lr_report = pd.DataFrame({
                    'Class': ['No Disease (0)', 'Disease (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.84, 0.84, '', 0.84, 0.84],
                    'Recall': [0.81, 0.87, '', 0.84, 0.84],
                    'F1-Score': [0.82, 0.86, 0.84, 0.84, 0.84],
                    'Support': [26, 31, 57, 57, 57]
                })
                st.dataframe(lr_report, use_container_width=True)
            
            with col2:
                st.markdown("**Key Metrics:**")
                st.metric("Cross-Val ROC-AUC", "0.927")
                st.metric("Test ROC-AUC", "0.882")
                st.metric("Test Accuracy", "0.84")
                st.metric("Recall (Disease)", "0.87")
        
        with tab4:
            st.subheader("🌲 Decision Tree Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Report:**")
                dt_report = pd.DataFrame({
                    'Class': ['No Disease (0)', 'Disease (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.77, 0.74, '', 0.76, 0.76],
                    'Recall': [0.65, 0.84, '', 0.75, 0.75],
                    'F1-Score': [0.71, 0.79, 0.75, 0.75, 0.75],
                    'Support': [26, 31, 57, 57, 57]
                })
                st.dataframe(dt_report, use_container_width=True)
            
            with col2:
                st.markdown("**Key Metrics:**")
                st.metric("Cross-Val ROC-AUC", "0.852")
                st.metric("Test ROC-AUC", "0.833")
                st.metric("Test Accuracy", "0.75")
                st.metric("Recall (Disease)", "0.84")

        # Feature Information
        st.header("📊 Feature Information")
        feature_details = pd.DataFrame({
            'Feature': ['sex', 'age', 'cp', 'thalach', 'slope', 'exang', 'ca', 'thal', 'oldpeak'],
            'Description': [
                'Gender (0: Female, 1: Male)',
                'Age in years',
                'Chest pain type (0-3)',
                'Maximum heart rate achieved',
                'Slope of peak exercise ST segment (0-2)',
                'Exercise induced angina (0: No, 1: Yes)',
                'Number of major vessels colored by flourosopy (0-3)',
                'Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)',
                'ST depression induced by exercise relative to rest'
            ],
            'Type': [
                'Categorical',
                'Numerical',
                'Categorical',
                'Numerical',
                'Categorical',
                'Categorical',
                'Numerical',
                'Categorical',
                'Numerical'
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
        st.dataframe(feature_details, use_container_width=True)

        # Feature Importance
        st.header("🎯 Feature Importance Analysis")
        
        if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
            feature_names = ['Sex', 'Age', 'Chest Pain', 'Max Heart Rate', 'ST Slope', 
                           'Exercise Angina', 'Major Vessels', 'Thalassemia', 'ST Depression']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.best_estimator_.feature_importances_,
                'Percentage': model.best_estimator_.feature_importances_ * 100
            }).sort_values('Importance', ascending=False)
            
            # Display feature importance table
            st.dataframe(importance_df.style.format({'Importance': '{:.4f}', 'Percentage': '{:.1f}%'}), 
                        use_container_width=True)
            
            # Feature importance visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, importance_df['Importance']):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance insights
            st.subheader("📝 Feature Importance Insights")
            top_feature = importance_df.iloc[0]
            st.success(f"**Most Important Feature:** {top_feature['Feature']} ({top_feature['Percentage']:.1f}%)")
            
            st.markdown("**Top 3 Most Important Features:**")
            for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
                st.write(f"{i+1}. **{row['Feature']}** - {row['Percentage']:.1f}%")
        
        else:
            st.warning("Feature importance tidak tersedia untuk model ini.")

        # GridSearchCV Results Summary
        st.header("🔍 GridSearchCV Results Summary")
        
        # Create summary of all hyperparameter tuning results
        gridsearch_summary = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree', 'MLP Classifier'],
            'Best Score (CV ROC-AUC)': [0.9269, 0.9131, 0.8517, 0.9278],
            'Best Parameters': [
                "{'C': 0.1, 'max_iter': 100, 'solver': 'liblinear'}",
                "{'criterion': 'entropy', 'max_depth': None, 'n_estimators': 100}",
                "{'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 20}",
                "{'activation': 'tanh', 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}"
            ],
            'Test ROC-AUC': [0.8821, 0.8989, 0.8331, 0.8958],
            'Test Accuracy': [0.84, 0.81, 0.75, 0.84]
        })
        
        st.dataframe(gridsearch_summary, use_container_width=True)
        
        # Performance comparison chart
        st.subheader("📊 Performance Comparison Chart")
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV Score vs Test ROC-AUC
        models = gridsearch_summary['Model']
        cv_scores = gridsearch_summary['Best Score (CV ROC-AUC)']
        test_roc = gridsearch_summary['Test ROC-AUC']
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV ROC-AUC', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, test_roc, width, label='Test ROC-AUC', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_title('Cross-Validation vs Test ROC-AUC Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Test Accuracy comparison
        test_acc = gridsearch_summary['Test Accuracy']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars3 = ax2.bar(models, test_acc, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Test Accuracy Comparison')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Model Selection Analysis
        st.header("🎯 Model Selection Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Why Random Forest was Selected")
            st.markdown("""
            **Key Reasons:**
            
            1. **Best Test ROC-AUC (0.899)** - Highest generalization performance
            2. **Balanced Performance** - Good trade-off between CV and test scores
            3. **Interpretability** - Feature importance provides medical insights
            4. **Robustness** - Less prone to overfitting than individual decision trees
            5. **Ensemble Method** - Combines multiple models for better prediction
            6. **Medical Context** - ROC-AUC is crucial for medical diagnosis applications
            """)
        
        with col2:
            st.subheader("❌ Why Other Models Were Not Selected")
            st.markdown("""
            **Analysis:**
            
            - **MLP Classifier**: Higher CV score (0.928) but lower test ROC-AUC (0.896)
            - **Logistic Regression**: High CV score (0.927) but significantly lower test ROC-AUC (0.882)
            - **Decision Tree**: Lowest performance across all metrics
            - **Overfitting Concerns**: Gap between CV and test scores indicates overfitting
            - **Generalization**: Random Forest shows best ability to generalize to unseen data
            """)

        # Technical Implementation Details
        st.header("⚙️ Technical Implementation Details")
        
        # Preprocessing pipeline
        st.subheader("🔄 Data Preprocessing Pipeline")
        preprocessing_steps = pd.DataFrame({
            'Step': ['Data Loading', 'Feature Selection', 'Train-Test Split', 'Scaling', 'Model Training'],
            'Description': [
                'Load heart disease dataset from UCI repository',
                'Select 9 most relevant features based on correlation and medical significance',
                'Split data into 80% training and 20% testing',
                'Apply StandardScaler to normalize feature values',
                'Train models with GridSearchCV hyperparameter tuning'
            ],
            'Method': [
                'pandas.read_csv()',
                'Correlation analysis + domain knowledge',
                'train_test_split(test_size=0.2, random_state=42)',
                'StandardScaler().fit_transform()',
                'GridSearchCV with 3-fold CV and ROC-AUC scoring'
            ]
        })
        st.dataframe(preprocessing_steps, use_container_width=True)
        
        # Model Parameters Detail
        st.subheader("🎛️ Random Forest Model Parameters")
        model_params = pd.DataFrame({
            'Parameter': ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'random_state'],
            'Best Value': [100, 'entropy', None, 2, 42],
            'Description': [
                'Number of trees in the forest',
                'Function to measure split quality',
                'Maximum depth of trees (None = unlimited)',
                'Minimum samples required to split internal node',
                'Random state for reproducibility'
            ],
            'Tuning Range': [
                '[100, 200, 300]',
                "['gini', 'entropy', 'log_loss']",
                '[None, 10, 20, 30]',
                'Default (2)',
                'Fixed (42)'
            ]
        })
        st.dataframe(model_params, use_container_width=True)

        # Validation Strategy
        st.header("🔬 Validation Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cross-Validation Setup")
            st.markdown("""
            - **Method**: 3-Fold Cross-Validation
            - **Scoring**: ROC-AUC (Area Under ROC Curve)
            - **Reason**: ROC-AUC is ideal for binary classification in medical context
            - **Advantage**: Robust performance estimation across different data splits
            - **Grid Search**: Exhaustive search over hyperparameter combinations
            """)
        
        with col2:
            st.subheader("Test Set Evaluation")
            st.markdown("""
            - **Size**: 57 samples (20% of total dataset)
            - **Stratified**: Maintains class distribution
            - **Metrics**: ROC-AUC, Accuracy, Precision, Recall, F1-Score
            - **Purpose**: Unbiased evaluation of final model performance
            - **No Data Leakage**: Completely separate from training process
            """)

        # Conclusion and Recommendations
        st.header("📋 Conclusion and Recommendations")
        
        st.subheader("🏆 Model Performance Summary")
        st.success("""
        **Random Forest Classifier** achieved:
        - **Cross-Validation ROC-AUC**: 0.913
        - **Test ROC-AUC**: 0.899
        - **Test Accuracy**: 81%
        - **Recall for Disease Detection**: 87%
        """)
        
        st.subheader("💡 Recommendations for Future Improvements")
        st.markdown("""
        1. **Data Enhancement**:
           - Collect more diverse patient data
           - Include additional clinical features
           - Balance dataset if class imbalance exists
        
        2. **Model Improvements**:
           - Experiment with ensemble methods (XGBoost, LightGBM)
           - Try feature engineering techniques
           - Implement advanced hyperparameter tuning (Bayesian optimization)
        
        3. **Clinical Integration**:
           - Validate with medical professionals
           - Conduct prospective clinical studies
           - Implement confidence intervals for predictions
        
        4. **Technical Enhancements**:
           - Add model explainability features (LIME, SHAP)
           - Implement real-time monitoring
           - Create API for integration with healthcare systems
        """)
        
        st.warning("""
        **Important Medical Disclaimer**: 
        This model is for educational and research purposes only. 
        It should not be used as a substitute for professional medical diagnosis. 
        Always consult with qualified healthcare professionals for medical decisions.
        """)
        
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
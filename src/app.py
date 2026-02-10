"""
Credit Card Default Classifier - Streamlit Application

This application allows users to upload test data, select trained classification
models, and view predictions along with evaluation metrics and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, Optional, Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from model_utils import load_model, load_metrics as load_metrics_from_file

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Model configuration
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
METRICS_PATH = os.path.join(PROJECT_ROOT, "model", "metrics.json")
SCALER_PATH = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data.csv")

# Target column name (label)
TARGET_COLUMN = 'dpnm'

# Required feature columns for CSV validation
REQUIRED_FEATURE_COLUMNS = [
    'LIMIT_BAL',
    'SEX',
    'EDUCATION',
    'MARRIAGE',
    'AGE',
    'PAY_1',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6',
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6',
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6',
]

# Page configuration - centered layout with collapsed sidebar
st.set_page_config(
    page_title="Credit Card Default Classifier",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load and apply custom CSS
CSS_PATH = os.path.join(SCRIPT_DIR, "styles.css")

def load_css() -> str:
    with open(CSS_PATH, "r") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_css(), unsafe_allow_html=True)

@st.cache_resource
def load_scaler() -> Optional[StandardScaler]:
    try:
        return joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.session_state['scaler_load_error'] = f"Scaler file not found at {SCALER_PATH}"
        return None
    except Exception as e:
        st.session_state['scaler_load_error'] = f"Error loading scaler: {str(e)}"
        return None


@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    models = {}
    errors = []
    
    for model_name in MODEL_NAMES.keys():
        try:
            models[model_name] = load_model(model_name, MODEL_DIR)
        except FileNotFoundError:
            models[model_name] = None
            errors.append(f"Model file not found: {model_name}.pkl")
        except Exception as e:
            models[model_name] = None
            errors.append(f"Error loading {model_name}: {str(e)}")
    
    if errors:
        st.session_state['model_load_errors'] = errors
    
    return models


@st.cache_data
def load_metrics() -> Optional[Dict[str, Dict[str, float]]]:
    try:
        return load_metrics_from_file(METRICS_PATH)
    except FileNotFoundError:
        st.session_state['metrics_load_error'] = f"Metrics file not found at {METRICS_PATH}"
        return None
    except Exception as e:
        st.session_state['metrics_load_error'] = f"Error loading metrics: {str(e)}"
        return None


@st.cache_data
def load_sample_data() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(TEST_DATA_PATH)
    except Exception:
        return None

def validate_uploaded_data(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    if df is None or df.empty:
        return False, "Uploaded file is empty or could not be read.", []
    
    missing_columns = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        return False, error_msg, missing_columns
    
    return True, "", []

def display_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1E2130', '#0D4F6B', '#0077A3', '#00A3CC', '#00D4FF']
    custom_cmap = LinearSegmentedColormap.from_list('dark_cyan', colors)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=custom_cmap,
        xticklabels=['No Default (0)', 'Default (1)'],
        yticklabels=['No Default (0)', 'Default (1)'],
        ax=ax,
        cbar_kws={'label': 'Count'},
        annot_kws={'color': 'white', 'fontsize': 16, 'fontweight': 'bold'},
        linewidths=2,
        linecolor='#0E1117'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, color='#00D4FF')
    ax.set_ylabel('True Label', fontsize=12, color='#00D4FF')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#00D4FF')
    ax.tick_params(colors='#00D4FF')
    
    ax.set_xticklabels(ax.get_xticklabels(), color='#00D4FF')
    ax.set_yticklabels(ax.get_yticklabels(), color='#00D4FF')
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('#00D4FF')
    cbar.ax.tick_params(colors='#00D4FF')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=['No Default (0)', 'Default (1)'],
        output_dict=True
    )
    
    class_metrics = {
        'Class': ['No Default (0)', 'Default (1)'],
        'Precision': [
            report['No Default (0)']['precision'],
            report['Default (1)']['precision']
        ],
        'Recall': [
            report['No Default (0)']['recall'],
            report['Default (1)']['recall']
        ],
        'F1-Score': [
            report['No Default (0)']['f1-score'],
            report['Default (1)']['f1-score']
        ],
        'Support': [
            int(report['No Default (0)']['support']),
            int(report['Default (1)']['support'])
        ]
    }
    
    class_df = pd.DataFrame(class_metrics)
    
    for col in ['Precision', 'Recall', 'F1-Score']:
        class_df[col] = class_df[col].apply(lambda x: f"{x:.3f}")
    
    st.markdown('<p class="bright-label">Per-Class Metrics:</p>', unsafe_allow_html=True)
    st.dataframe(class_df, width='stretch', hide_index=True)
    
    summary_metrics = {
        'Metric': ['Accuracy', 'Macro Avg', 'Weighted Avg'],
        'Precision': [
            '-',
            f"{report['macro avg']['precision']:.3f}",
            f"{report['weighted avg']['precision']:.3f}"
        ],
        'Recall': [
            '-',
            f"{report['macro avg']['recall']:.3f}",
            f"{report['weighted avg']['recall']:.3f}"
        ],
        'F1-Score': [
            f"{report['accuracy']:.3f}",
            f"{report['macro avg']['f1-score']:.3f}",
            f"{report['weighted avg']['f1-score']:.3f}"
        ],
        'Support': [
            int(report['macro avg']['support']),
            int(report['macro avg']['support']),
            int(report['weighted avg']['support'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_metrics)
    
    st.markdown('<p class="bright-label">Summary Metrics:</p>', unsafe_allow_html=True)
    st.dataframe(summary_df, width='stretch', hide_index=True)


def create_results_dataframe(
    original_data: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> pd.DataFrame:
    results_df = original_data.copy()
    results_df['Prediction'] = predictions
    results_df['Prediction_Label'] = results_df['Prediction'].map({
        0: 'No Default',
        1: 'Default'
    })
    
    if probabilities is not None:
        results_df['Default_Probability'] = probabilities[:, 1]
    
    return results_df


def display_results_table(results_df: pd.DataFrame) -> None:
    # Define columns to display (prioritize prediction columns first for visibility)
    display_columns = ['Prediction_Label', 'Prediction']
    
    if 'Default_Probability' in results_df.columns:
        display_columns.append('Default_Probability')
    
    # Add original feature columns
    feature_columns = [col for col in results_df.columns if col not in display_columns]
    display_columns.extend(feature_columns)
    
    # Reorder DataFrame for display
    display_df = results_df[display_columns].copy()
    
    # Apply styling function for prediction labels
    def style_prediction_label(val):
        if val == 'Default':
            return 'color: #FF6B6B; font-weight: bold'
        elif val == 'No Default':
            return 'color: #00FF88; font-weight: bold'
        return ''
    
    # Create styled dataframe
    styled_df = display_df.style.applymap(
        style_prediction_label,
        subset=['Prediction_Label']
    )
    
    # Format probability column if present
    if 'Default_Probability' in display_df.columns:
        styled_df = styled_df.format({'Default_Probability': '{:.3f}'})
    
    st.dataframe(styled_df, width='stretch', height=400)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Load models, metrics, and scaler
models = load_all_models()
metrics = load_metrics()
scaler = load_scaler()

# Get available models (those that loaded successfully)
available_models = {name: display for name, display in MODEL_NAMES.items() 
                   if models.get(name) is not None}

# =============================================================================
# 1. HEADER SECTION
# =============================================================================

st.markdown('<p class="app-header">💳 Credit Card Default Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="app-description">Predict credit card defaults using machine learning models trained on the Kaggle dataset</p>', unsafe_allow_html=True)

# =============================================================================
# 2. UPLOAD DATA SECTION
# =============================================================================

st.markdown('<p class="section-header">📤 Upload Data</p>', unsafe_allow_html=True)

# Sample data download
sample_data = load_sample_data()
if sample_data is not None:
    st.download_button(
        label="📥 Download Sample CSV (test_data.csv)",
        data=sample_data.to_csv(index=False),
        file_name="test_data.csv",
        mime="text/csv",
        help="Download a sample CSV file to see the required format"
    )

# File uploader
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file containing credit card client features for prediction"
)

# Process uploaded file
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        is_valid, error_msg, missing_cols = validate_uploaded_data(uploaded_df)
        
        if is_valid:
            # Extract features (exclude label column if present)
            features_df = uploaded_df[REQUIRED_FEATURE_COLUMNS].copy()
            
            # Check if label column is present and store it
            if TARGET_COLUMN in uploaded_df.columns:
                st.session_state['uploaded_labels'] = uploaded_df[TARGET_COLUMN].values
            else:
                st.session_state['uploaded_labels'] = None
            
            # Apply scaling if scaler is available
            if scaler is not None:
                scaled_features = scaler.transform(features_df)
                st.session_state['uploaded_data'] = scaled_features
                st.session_state['uploaded_data_unscaled'] = features_df
            else:
                st.warning("Scaler not found. Using unscaled data (predictions may be inaccurate).")
                st.session_state['uploaded_data'] = features_df.values
                st.session_state['uploaded_data_unscaled'] = features_df
            
            st.session_state['upload_valid'] = True
            st.session_state['upload_error'] = None
            st.success(f"File uploaded successfully!")
            
            # Display row count and column count info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric(label="Rows", value=len(uploaded_df))
            with info_col2:
                st.metric(label="Features", value=len(REQUIRED_FEATURE_COLUMNS))
            with info_col3:
                has_labels = TARGET_COLUMN in uploaded_df.columns
                st.metric(label="Labels", value="Yes" if has_labels else "No")
            
            # Data preview - first 5 rows
            st.markdown('<p class="bright-label">Data Preview (first 5 rows):</p>', unsafe_allow_html=True)
            preview_rows = min(5, len(uploaded_df))
            st.dataframe(uploaded_df.head(preview_rows), width='stretch')
        else:
            st.session_state['uploaded_data'] = None
            st.session_state['uploaded_data_unscaled'] = None
            st.session_state['uploaded_labels'] = None
            st.session_state['upload_valid'] = False
            st.session_state['upload_error'] = error_msg
            
            # Display validation error with missing columns
            st.error(f"Invalid CSV file")
            st.markdown("**Missing required columns:**")
            if missing_cols:
                for col in missing_cols:
                    st.markdown(f"- `{col}`")
            else:
                st.markdown(error_msg)
            
    except Exception as e:
        st.session_state['uploaded_data'] = None
        st.session_state['uploaded_data_unscaled'] = None
        st.session_state['uploaded_labels'] = None
        st.session_state['upload_valid'] = False
        st.session_state['upload_error'] = f"Error reading file: {str(e)}"
        st.error(f"Error reading file: {str(e)}")
else:
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None
        st.session_state['uploaded_data_unscaled'] = None
        st.session_state['uploaded_labels'] = None
        st.session_state['upload_valid'] = False
        st.session_state['upload_error'] = None

st.markdown("---")

# =============================================================================
# 3. MODEL SELECTION & PREDICTION SECTION
# =============================================================================

st.markdown('<p class="section-header">🤖 Model Selection & Prediction</p>', unsafe_allow_html=True)

# Only show model selection if data is uploaded
if st.session_state.get('upload_valid') and st.session_state.get('uploaded_data') is not None:
    if available_models:
        st.markdown('<p class="bright-label">Select a model:</p>', unsafe_allow_html=True)
        selected_model_display = st.selectbox(
            "Select a model:",
            options=list(available_models.values()),
            index=0,
            label_visibility="collapsed"
        )
        selected_model = [k for k, v in MODEL_NAMES.items() if v == selected_model_display][0]
        st.session_state['selected_model'] = selected_model
        
        # Generate Predictions button
        if st.button("🔮 Generate Predictions", type="primary"):
            try:
                model = models[selected_model]
                data = st.session_state['uploaded_data']
                
                with st.spinner("Generating predictions..."):
                    # Data is already scaled (numpy array), use directly
                    if isinstance(data, np.ndarray):
                        predictions = model.predict(data)
                    else:
                        predictions = model.predict(data.values)
                    
                    try:
                        if isinstance(data, np.ndarray):
                            probabilities = model.predict_proba(data)
                        else:
                            probabilities = model.predict_proba(data.values)
                        st.session_state['prediction_probabilities'] = probabilities
                    except Exception:
                        st.session_state['prediction_probabilities'] = None
                
                st.session_state['predictions'] = predictions
                st.session_state['prediction_model'] = selected_model
                st.session_state['prediction_error'] = None
                
                st.success(f"Generated {len(predictions)} predictions successfully!")
                
            except Exception as e:
                st.session_state['predictions'] = None
                st.session_state['prediction_model'] = None
                st.session_state['prediction_error'] = str(e)
                st.error(f"Error generating predictions: {str(e)}")
    else:
        st.error("No models available. Please train models first.")
        selected_model = None
else:
    st.info("📁 Please upload a valid CSV file to select a model and generate predictions.")
    selected_model = st.session_state.get('selected_model')

st.markdown("---")

# =============================================================================
# 4. MODEL PERFORMANCE METRICS SECTION
# =============================================================================

st.markdown('<p class="section-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)

if st.session_state.get('predictions') is not None and st.session_state.get('prediction_error') is None:
    prediction_model = st.session_state.get('prediction_model')
    
    if prediction_model and metrics:
        model_metrics = metrics.get(prediction_model)
        
        if model_metrics:
            st.markdown('<p class="bright-label">Model: {}</p>'.format(MODEL_NAMES.get(prediction_model, prediction_model)), unsafe_allow_html=True)
            
            # Row 1: Accuracy, AUC Score, Precision
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = model_metrics.get('accuracy', 0)
                st.metric(label="Accuracy", value=f"{accuracy:.3f}")
            
            with col2:
                auc_score = model_metrics.get('auc_score', 0)
                st.metric(label="AUC Score", value=f"{auc_score:.3f}")
            
            with col3:
                precision = model_metrics.get('precision', 0)
                st.metric(label="Precision", value=f"{precision:.3f}")
            
            # Row 2: Recall, F1 Score, MCC
            col4, col5, col6 = st.columns(3)
            
            with col4:
                recall = model_metrics.get('recall', 0)
                st.metric(label="Recall", value=f"{recall:.3f}")
            
            with col5:
                f1_score = model_metrics.get('f1_score', 0)
                st.metric(label="F1 Score", value=f"{f1_score:.3f}")
            
            with col6:
                mcc_score = model_metrics.get('mcc_score', 0)
                st.metric(label="MCC Score", value=f"{mcc_score:.3f}")
            
            # Confusion Matrix and Classification Report
            # Use uploaded labels from the uploaded file
            uploaded_labels = st.session_state.get('uploaded_labels')
            predictions = st.session_state['predictions']
            
            if uploaded_labels is not None and len(uploaded_labels) == len(predictions):
                st.markdown("---")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.markdown('<p class="bright-label">Confusion Matrix</p>', unsafe_allow_html=True)
                    display_confusion_matrix(uploaded_labels, predictions)
                
                with viz_col2:
                    st.markdown('<p class="bright-label">Classification Report</p>', unsafe_allow_html=True)
                    display_classification_report(uploaded_labels, predictions)
            else:
                st.info("📊 Upload data with the 'dpnm' label column to see confusion matrix and classification report.")
        else:
            st.warning(f"No metrics available for {MODEL_NAMES.get(prediction_model, prediction_model)}")
else:
    st.info("Generate predictions to view model performance metrics.")

st.markdown("---")

# =============================================================================
# 5. RESULTS SECTION
# =============================================================================

st.markdown('<p class="section-header">📋 Results</p>', unsafe_allow_html=True)

if st.session_state.get('predictions') is not None and st.session_state.get('prediction_error') is None:
    predictions = st.session_state['predictions']
    prediction_model = st.session_state.get('prediction_model')
    
    # Summary statistics
    col_pred1, col_pred2, col_pred3 = st.columns(3)
    
    default_count = sum(predictions == 1)
    no_default_count = sum(predictions == 0)
    default_rate = (default_count / len(predictions)) * 100 if len(predictions) > 0 else 0
    
    with col_pred1:
        st.metric("Total Predictions", len(predictions))
    with col_pred2:
        st.metric("Predicted Defaults", default_count)
    with col_pred3:
        st.metric("Default Rate", f"{default_rate:.1f}%")
    
    # Results table
    st.markdown('<p class="bright-label">Prediction Results:</p>', unsafe_allow_html=True)
    
    # Create results DataFrame with original (unscaled) data and predictions
    # Use unscaled data for display and download
    unscaled_data = st.session_state.get('uploaded_data_unscaled')
    if unscaled_data is not None:
        results_df = create_results_dataframe(
            unscaled_data,
            predictions,
            st.session_state.get('prediction_probabilities')
        )
    else:
        # Fallback to scaled data if unscaled not available
        data = st.session_state['uploaded_data']
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=REQUIRED_FEATURE_COLUMNS)
        results_df = create_results_dataframe(
            data,
            predictions,
            st.session_state.get('prediction_probabilities')
        )
    
    # Store results in session state for download
    st.session_state['results_df'] = results_df
    
    # Display styled results table
    display_results_table(results_df)
    
    # Download button with descriptive text
    st.markdown("---")
    st.markdown('<p class="bright-label">Export Results:</p>', unsafe_allow_html=True)
    csv = results_df.to_csv(index=False)
    model_display_name = MODEL_NAMES.get(prediction_model, prediction_model)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"predictions_{prediction_model}.csv",
        mime="text/csv",
        help=f"Download predictions from {model_display_name} model with original data"
    )
else:
    st.info("📊 Generate predictions to view and download results.")

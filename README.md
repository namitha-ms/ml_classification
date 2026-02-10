# Credit Card Default Classifier

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a credit card client will default on their payment in the next month. 

## b. Dataset Description

The dataset used is the **UCI Credit Card Default Dataset** (also known as "Default of Credit Card Clients Dataset").

| Attribute | Description |
|-----------|-------------|
| **Source** | [Kaggle - Default of Credit Card Clients](https://www.kaggle.com/datasets/mariosfish/default-of-credit-card-clients) |
| **Samples** | 30,000 credit card clients |
| **Features** | 23 input features |
| **Target** | Binary (0 = No Default, 1 = Default) |
| **Time Period** | April 2005 to September 2005 (Taiwan) |

### Feature Description

| Feature | Description |
|---------|-------------|
| LIMIT_BAL | Amount of credit given (NT dollar) |
| SEX | Gender (1 = male, 2 = female) |
| EDUCATION | Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others) |
| MARRIAGE | Marital status (1 = married, 2 = single, 3 = others) |
| AGE | Age in years |
| PAY_1 to PAY_6 | Repayment status from April to September 2005 (-2 = no consumption, -1 = paid in full, 0 = revolving credit, 1-9 = months of payment delay) |
| BILL_AMT1 to BILL_AMT6 | Bill statement amount from April to September 2005 (NT dollar) |
| PAY_AMT1 to PAY_AMT6 | Previous payment amount from April to September 2005 (NT dollar) |
| dpnm | Default payment next month (0 = No, 1 = Yes) - **Target Variable** |

## c. Models Used

Six classification models were implemented and compared:

1. **Logistic Regression** - A linear model that estimates the probability of default using a logistic function
2. **Decision Tree** - A tree-based model that makes decisions based on feature thresholds
3. **K-Nearest Neighbors (kNN)** - An instance-based model that classifies based on the majority class of k nearest neighbors
4. **Naive Bayes** - A probabilistic classifier based on Bayes' theorem with independence assumptions
5. **Random Forest (Ensemble)** - An ensemble of decision trees using bagging to improve accuracy and reduce overfitting
6. **XGBoost (Ensemble)** - A gradient boosting ensemble method known for high performance on structured data

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8100 | 0.7270 | 0.6927 | 0.2369 | 0.3530 | 0.3259 |
| Decision Tree | 0.7210 | 0.6069 | 0.3730 | 0.4037 | 0.3877 | 0.2077 |
| kNN | 0.7950 | 0.7077 | 0.5487 | 0.3564 | 0.4321 | 0.3247 |
| Naive Bayes | 0.7070 | 0.7371 | 0.3967 | 0.6504 | 0.4928 | 0.3218 |
| Random Forest (Ensemble) | 0.8163 | 0.7574 | 0.6405 | 0.3663 | 0.4661 | 0.3857 |
| XGBoost (Ensemble) | 0.8148 | 0.7750 | 0.6347 | 0.3625 | 0.4615 | 0.3801 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves high accuracy (81%) but suffers from low recall (23.7%), indicating it misses many actual defaulters. The high precision (69.3%) means when it predicts default, it's often correct. Best suited when false positives are costly. |
| Decision Tree | Lowest overall performance with accuracy of 72.1% and AUC of 0.61. Shows signs of overfitting with poor generalization. The balanced but low precision/recall suggests the model struggles to find meaningful patterns. |
| kNN | Moderate performance with 79.5% accuracy. Better recall than Logistic Regression (35.6%) but lower precision (54.9%). Sensitive to the choice of k and feature scaling. Performance limited by the curse of dimensionality with 23 features. |
| Naive Bayes | Lowest accuracy (70.7%) but highest recall (65%) among all models, making it effective at identifying actual defaulters. The independence assumption limits precision (39.7%). Best choice when catching all potential defaults is critical. |
| Random Forest (Ensemble) | Best overall accuracy (81.6%) and strong AUC (0.76). Good balance between precision (64%) and recall (36.6%). The ensemble approach reduces overfitting seen in single Decision Tree. Highest MCC (0.39) indicates best overall classification quality. |
| XGBoost (Ensemble) | Second-best accuracy (81.5%) and highest AUC (0.78), indicating excellent ranking ability. Similar performance to Random Forest with slightly better probability calibration. Gradient boosting effectively captures complex feature interactions. |

## Key Insights

1. **Class Imbalance Impact**: The dataset has imbalanced classes (more non-defaulters than defaulters), which affects all models' ability to detect defaults (low recall across models).

2. **Ensemble Methods Excel**: Random Forest and XGBoost consistently outperform single models, demonstrating the power of ensemble learning for this problem.

3. **Precision-Recall Trade-off**: Models with high accuracy (Logistic Regression, Random Forest, XGBoost) tend to have lower recall, while Naive Bayes sacrifices accuracy for better default detection.

4. **Best Model Selection**:
   - For **overall performance**: Random Forest or XGBoost
   - For **catching more defaults** (high recall): Naive Bayes
   - For **confident predictions** (high precision): Logistic Regression

## Project Structure

```
├── data/
│   ├── credit_card_default.csv    # Original dataset
│   └── test_data.csv              # Test data with labels (unscaled)
├── model/
│   ├── logistic_regression.pkl    # Trained models
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl                 # StandardScaler for feature scaling
│   └── metrics.json               # Model evaluation metrics
├── src/
│   ├── app.py                     # Streamlit web application
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── model_utils.py             # Model utilities
│   ├── train_all_models.py        # Training script
│   ├── validate_models.py         # Model validation script
│   └── models/                    # Model class implementations
└── README.md
```

## Running the Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train models (if not already trained):
   ```bash
   python -m src.train_all_models
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Usage

1. Upload a CSV file with the 23 required feature columns
2. Select a classification model from the dropdown
3. Click "Generate Predictions" to get default predictions
4. View model performance metrics and confusion matrix
5. Download results as CSV

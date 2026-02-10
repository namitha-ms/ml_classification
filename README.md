# Credit Card Default Classifier

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a credit card client will default on their payment in the next month. 

## Dataset Description

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

## Models Used

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
| Logistic Regression | 0.6925 | 0.7276 | 0.3799 | 0.6405 | 0.4769 | 0.2978 |
| Decision Tree | 0.7202 | 0.7616 | 0.4113 | 0.6458 | 0.5025 | 0.3368 |
| kNN | 0.8087 | 0.7328 | 0.6194 | 0.3260 | 0.4271 | 0.3495 |
| Naive Bayes | 0.7070 | 0.7371 | 0.3967 | 0.6504 | 0.4928 | 0.3218 |
| Random Forest (Ensemble) | 0.7980 | 0.7759 | 0.5378 | 0.5476 | 0.5426 | 0.4131 |
| XGBoost (Ensemble) | 0.7648 | 0.7610 | 0.4702 | 0.5887 | 0.5228 | 0.3735 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves 69.3% accuracy with strong recall (64%), making it effective at identifying defaulters. Lower precision (38%) means more false positives, but the high recall ensures fewer actual defaulters are missed. Best suited when the cost of missing a defaulter outweighs false alarms. |
| Decision Tree | Solid AUC (0.76) and highest recall (64.6%) among tree-based models. F1 score of 0.50 indicates reasonable balance between precision and recall. Interpretable model that provides clear decision rules for understanding default patterns. |
| kNN | Highest accuracy (80.9%) and precision (62%) but lowest recall (32.6%). Conservative predictions — when it flags a default, it's usually correct, but misses two-thirds of actual defaulters. Best when false positives are costly and you can afford to miss some defaults. |
| Naive Bayes | Highest recall (65%) among all models, catching the most defaulters. Trades accuracy (70.7%) for detection capability. The probabilistic approach works well despite the independence assumption. Ideal when identifying all potential defaults is the priority. |
| Random Forest (Ensemble) | Best overall performer with highest AUC (0.78), MCC (0.41), and F1 score (0.54). Achieves the best balance between precision (54%) and recall (55%). The ensemble approach provides robust predictions. Recommended as the primary model for production use. |
| XGBoost (Ensemble) | Strong all-around performance with good recall (59%) and solid AUC (0.76). F1 score of 0.52 demonstrates effective balance. Gradient boosting captures complex feature interactions well. Second-best choice for balanced, reliable predictions. |

## Key Insights

1. **Ensemble Methods Excel**: Random Forest and XGBoost consistently outperform single models, with Random Forest achieving the best overall balance (highest MCC of 0.41 and F1 of 0.54).

2. **Precision-Recall Trade-off**: Models fall into two categories — high-recall models (Naive Bayes, Logistic Regression, Decision Tree) that catch more defaulters but generate more false alarms, and high-precision models (KNN) that are more conservative but miss more defaults.

3. **AUC as Ranking Metric**: Random Forest (0.78) and XGBoost (0.76) have the best AUC scores, indicating superior ability to rank customers by default risk.

4. **Best Model Selection**:
   - For **overall performance**: Random Forest (best MCC, F1, and AUC balance)
   - For **catching more defaults** (high recall): Naive Bayes (65%) or Logistic Regression (64%)
   - For **confident predictions** (high precision): KNN (62%)
   - For **balanced precision-recall**: Random Forest (54%/55%) or XGBoost (47%/59%)

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

# Customer-Churn-Prediction-for-Telecom-Customer-Retention

## Project Description
This project utilizes machine learning techniques to predict customer churn in a telecom company. By analyzing historical customer data, the goal is to build a predictive model that identifies customers at risk of leaving. This enables the company to implement proactive retention strategies, improving customer retention rates and overall business sustainability.

The dataset includes various features such as customer demographics, service usage, and account information. Through data exploration, feature engineering, and model evaluation, the project aims to uncover patterns that influence churn, leading to more effective decision-making.

## Problem the Model Solves
Customer churn is a significant challenge for telecom companies. Retaining existing customers is more cost-effective than acquiring new ones. By predicting churn, the company can take timely actions to retain high-risk customers, thereby optimizing customer lifetime value and reducing churn rates.

## Main Objectives
- Analyze key factors (demographics, service usage, account info) influencing customer churn.
- Build a predictive model to forecast customer churn.
- Implement targeted retention strategies for customers predicted to churn.
- Improve business sustainability by reducing churn.

---

## Table of Contents
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Summary](#summary)

---

## Installation Instructions
To get started with this project, follow these steps:

### Requirements:
- Python 3.x (preferably Python 3.7+)

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies:
- pandas (for data manipulation)
- numpy (for numerical operations)
- scikit-learn (for machine learning algorithms)
- xgboost (for gradient boosting model)
- lightgbm (for gradient boosting model)
- decision-tree (for decision tree model)
- random-forest (for random forest model)
- matplotlib (for data visualization)
- seaborn (for advanced visualization)
- jupyter (optional, for running notebooks)

---

## Usage
### Open the Jupyter Notebook:
Start the notebook by running:
```bash
jupyter notebook
```

Then, open the project notebook.

### Train the Model:
Run the cells in the notebook to preprocess the data and train the model.

### Make Predictions:
After training, use the notebook to make predictions on the test data.

### Evaluate the Model:
Evaluate model performance using appropriate metrics provided in the notebook.

---

## Data
The dataset used in this project is `Telco-Customer-Churn.csv` and contains customer information relevant to predicting churn.

### Features:
- **customerID**: Unique identifier for each customer
- **gender**: Gender of the customer
- **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)
- **tenure**: Length of time the customer has been with the company
- **PhoneService**: Whether the customer has phone service (Yes/No)
- **MultipleLines**: Whether the customer has multiple lines (Yes/No)
- **InternetService**: Type of internet service the customer uses
- **OnlineSecurity**: Whether the customer has online security (Yes/No)
- **OnlineBackup**: Whether the customer has online backup (Yes/No)
- **DeviceProtection**: Whether the customer has device protection (Yes/No)
- **TechSupport**: Whether the customer has tech support (Yes/No)
- **StreamingTV**: Whether the customer has streaming TV service (Yes/No)
- **StreamingMovies**: Whether the customer has streaming movie service (Yes/No)
- **Contract**: Type of contract (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether the customer uses paperless billing (Yes/No)
- **PaymentMethod**: Payment method used by the customer
- **MonthlyCharges**: Monthly charges billed to the customer
- **TotalCharges**: Total charges for the customer
- **Churn**: Target variable (1: Churned, 0: Did not churn)

### Data Preprocessing:
- Handle missing values and outliers
- Perform feature engineering (e.g., one-hot encoding for categorical variables)
- Normalize/scale features as needed
- Split data into training and testing sets

---

## Model Architecture
We use several machine learning models to predict customer churn:

- **Logistic Regression**: A simple linear model for binary classification
- **Decision Tree**: A non-linear model that can capture complex relationships
- **Random Forest Classifier**: An ensemble model combining decision trees to reduce overfitting
- **XGBoost**: A gradient-boosting algorithm known for handling imbalanced data
- **LightGBM**: A fast and efficient gradient boosting model

### Hyperparameters:
- **Logistic Regression**: Class weights for handling imbalanced classes
- **Decision Tree**: Maximum depth, class weights
- **Random Forest**: Number of trees, class weights
- **XGBoost**: Learning rate, max depth, subsample, etc.
- **LightGBM**: Learning rate, number of leaves, etc.

---

## Evaluation Metrics
Model performance is evaluated using the following metrics:

- **Recall (Sensitivity)**: Proportion of actual churned customers correctly identified
- **F1-Score**: Balances precision and recall, especially useful for imbalanced datasets
- **AUC-ROC**: Measures the model’s ability to distinguish between churned and non-churned customers

### Baseline Comparison:
Logistic Regression serves as the baseline model for comparison against more complex models like Random Forest, XGBoost, and LightGBM.

---

## Results
### Model Performance:

| Model                | Accuracy | Precision | Recall  | F1-Score | AUC     |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression   | 73.89%   | 50.56%    | 80.21%  | 62.03%   | 83.84%  |
| Decision Tree         | 70.52%   | 46.87%    | 81.28%  | 59.45%   | 81.79%  |
| Random Forest         | 75.59%   | 52.85%    | 75.94%  | 62.33%   | 83.79%  |
| XGBoost              | 57.73%   | 38.10%    | 94.47%  | 54.30%   | 82.05%  |
| LightGBM             | 67.01%   | 44.09%    | 89.84%  | 59.15%   | 83.95%  |

### Best Performing Model:
- **XGBoost** achieved the highest recall (94.47%), which is crucial for minimizing false negatives and identifying churned customers.

### Second Best Model:
- **LightGBM** offers a strong balance of recall, precision, and AUC, making it a strong candidate for practical deployment.

---

## Summary
This project aims to predict customer churn for a telecom company and optimize retention efforts. After evaluating various models, the following conclusions were drawn:

- **XGBoost** is best for minimizing false negatives (high recall), though its precision may need tuning.
- **LightGBM** offers the best overall balance between recall, precision, and F1-score.
- **Random Forest** provides stable, general-purpose performance but doesn’t excel in recall.

For proactive retention, **LightGBM** is recommended due to its balanced performance, while **XGBoost** can be considered if capturing all churned customers is a priority.

---

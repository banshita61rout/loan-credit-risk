# 🏦 Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-Expert-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-black)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Project Overview

This project analyzes loan application data and builds a Machine Learning model to predict whether a loan application will be **Approved** or **Rejected** based on an applicant's financial and personal information.

---

## 🚀 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📊 Exploratory Data Analysis

The dataset was explored using:

- Data Cleaning
- Missing Value Analysis
- Univariate Analysis
- Bivariate Analysis
- Correlation Heatmap
- Outlier Detection
- Feature Relationship Analysis

---

## Key Insights

- Applicants with **higher Credit Scores** have a significantly higher chance of loan approval.
- **Applicant Income alone** is not a strong indicator of loan approval.
- Lower **Debt-to-Income (DTI) Ratio** is generally associated with approved loans.
- Applicants with **higher Savings and Collateral Value** tend to receive more approvals.
- Credit Score is one of the **strongest predictive features** in the dataset.

---
##  Machine Learning Models

- Built and evaluated **Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes** for loan approval prediction.
- Performed **data preprocessing**, including missing value handling, feature encoding, train-test split, and feature scaling.
- Conducted **EDA** to analyze income, credit score, DTI ratio, and their impact on loan approval.
- Applied **feature engineering** by creating transformed features from Credit Score and DTI Ratio to improve model performance.
- **Logistic Regression:** Balanced performance with **Precision: 78%** and **Recall: 77%**.
- **KNN:** Underperformed on this dataset with **Recall: 49%**, making it less suitable.
- **Naive Bayes:** Achieved the **highest Precision (80%)**, making it the preferred model when minimizing approvals of high-risk customers is the priority.
- **Final Outcome:** **Naive Bayes** selected as the best model based on precision, while **Logistic Regression** offered the best balance between precision and recall after feature engineering.
  
---

## 📂 Project Structure

```
Loan-Approval-Prediction/
│
├── dataset/
├── notebook.ipynb
├── README.md
└── .gitignore
```

---

## 📌 Future Improvements

- Hyperparameter Tuning
- Feature Engineering
- Model Comparison
- Deploy using Streamlit/Flask

---

### ⭐ If you found this project useful, consider giving it a Star!
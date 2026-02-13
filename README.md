# 🏦 Loan Prediction

This project presents a complete machine learning pipeline for predicting loan eligibility. The solution was developed as part of the **Analytics Vidhya Loan Prediction Practice Problem**.

🔗 Hackathon Link:
[https://www.analyticsvidhya.com/datahack/contest/practice-problem-loan-prediction-iii/](https://www.analyticsvidhya.com/datahack/contest/practice-problem-loan-prediction-iii/)

---

## 📌 Problem Statement

Dream Housing Finance provides home loans across urban, semi-urban, and rural areas. Customers apply online, and the company must assess their eligibility before approving the loan.

The objective of this project is to **automate the loan eligibility process in real-time** using customer information provided during the application process, including:

* Gender
* Marital Status
* Education
* Number of Dependents
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

Using historical data, we build predictive models to identify applicants likely to be approved for a loan.

---

## 💡 Hypothesis Generation

Based on domain knowledge, the following factors were hypothesised to influence loan approval:

* **Applicant Income:** Higher income increases approval likelihood.
* **Credit History:** Positive credit history strongly increases approval chances.
* **Loan Amount:** Smaller loan amounts are more likely to be approved.
* **Loan Term:** Shorter loan duration may improve approval probability.
* **EMI:** Lower monthly repayment burden increases chances of approval.

These hypotheses were tested using exploratory data analysis and machine learning models.

---

## ⚙️ Project Setup

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ .gitignore Includes

```
venv/
__pycache__/
.ipynb_checkpoints/
.env
.DS_Store
```

---

## 📊 Project Workflow

Notebook: `/code/data.ipynb`

### 🔍 1. Exploratory Data Analysis (EDA)

* Dataset structure & shape
* Data types
* Univariate analysis
* Bivariate analysis
* Missing value imputation
* Outlier treatment

### 🛠 2. Feature Engineering

Created new features based on financial reasoning:

* **Total Income**
* **EMI (Equated Monthly Instalment)**
* **Balance Income**
* Log transformations for skewed variables

### 🤖 3. Model Building

Models implemented:

* Logistic Regression (Baseline)
* Logistic Regression with Stratified K-Fold Cross Validation
* Decision Tree
* Random Forest
* XGBoost

### 📈 Model Comparison

| Model               | Validation Accuracy | Submission Accuracy |
| ------------------- | ------------------- | ------------------- |
| Logistic Regression | 0.796               | 0.792               |
| Decision Tree       | 0.707               | 0.653               |
| Random Forest       | 0.809               | 0.778               |
| XGBoost             | 0.749               | 0.722               |

### 🏆 Final Model Selection

**Logistic Regression** was selected as the final model due to:

* Strong and stable generalisation performance
* Minimal overfitting gap between validation and submission
* Simplicity and interpretability
* Computational efficiency

---

## 📁 Repository Structure

```
Loan_Prediction/
│
├── code/
│   ├── data.ipynb
│   ├── submission_logistic.csv
│   ├── submission_logistc_kfolds_validation.csv
│   ├── submission_logistic_2.csv
│   ├── submission_decision_tree.csv
│   ├── submission_random_forest.csv
│   ├── submission_xgboost.csv
│
├── datasets/
│   ├── sample_submission_49d68Cx.csv
│   ├── test_lAUu6dG.csv
│   ├── train_ctrUa4K.csv
│
├── Hypothesis_generation.txt
├── Problem_Statement.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📂 Submission Files

* `submission_logistic.csv` – Baseline Logistic Regression
* `submission_logistic_kfolds_validation.csv` – Logistic Regression with K-Folds
* `submission_logistic_2.csv` – Logistic Regression after feature engineering
* `submission_decision_tree.csv` – Decision Tree results
* `submission_random_forest.csv` – Random Forest results
* `submission_xgboost.csv` – XGBoost results

---

## 🚀 Key Learnings

* Feature engineering significantly improves model performance.
* Simpler models can outperform complex ensemble methods when properly tuned.
* Cross-validation provides more reliable performance estimation.
* Financial feature design (EMI, Balance Income) adds strong predictive value.

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost

---

## 📌 Author

Hamza Latif

---

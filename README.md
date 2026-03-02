# 🎮 Intelligent Player Churn Prediction  
### Machine Learning–Driven Retention Analytics

---

## 📌 Overview

Player retention is one of the most critical drivers of revenue in the online gaming industry. Even small increases in churn can significantly affect long-term profitability.

This project builds an end-to-end machine learning pipeline to predict player churn using behavioral and demographic data. The goal is to identify at-risk players early so companies can take proactive retention actions.

---

## 🎯 Problem Statement

Churn is defined as players exhibiting low engagement, indicating a high likelihood of discontinuing gameplay.

We formulated churn detection as a binary classification problem:

- **Churn = 1 → Low Engagement**
- **Churn = 0 → Medium/High Engagement**

The objective is to accurately distinguish between players likely to churn and those likely to remain active.

---

## 📊 Dataset Information

- **Total Records:** 40,034 players  
- **Total Features:** 13  
- **Target Variable:** Binary Churn Label  

### Key Features

**Demographic Features**
- Age  
- Gender  
- Location  

**Game & Behavioral Features**
- GameGenre  
- PlayTimeHours  
- SessionsPerWeek  
- AvgSessionDurationMinutes  
- PlayerLevel  
- AchievementsUnlocked  

> Behavioral engagement features were significantly more predictive than demographic attributes.

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Removed identifier column (`PlayerID`)
- Converted engagement levels into binary churn labels
- Applied One-Hot Encoding to categorical variables
- Standardized numerical features using `StandardScaler`
- Performed 80–20 stratified train-test split

### 🔹 Feature Engineering

Created a new composite metric:

```python
EngagementScore = SessionsPerWeek * AvgSessionDurationMinutes
```

This feature captures overall engagement intensity and improved predictive performance.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression (Baseline)
- Accuracy: ~87%
- ROC-AUC: ~0.90

### 2️⃣ Random Forest Classifier (Best Performing Model)
- Accuracy: ~92%
- ROC-AUC: ~0.94

Random Forest outperformed Logistic Regression due to its ability to capture nonlinear relationships and feature interactions.

---

## 📈 Evaluation Metrics

- Accuracy  
- ROC-AUC Score  
- Precision  
- Recall  
- Confusion Matrix  

A ROC-AUC above **0.90** indicates strong class separation capability.

---

## 🔍 Key Insights

- SessionsPerWeek is the strongest churn predictor.
- Engagement duration significantly influences retention.
- Behavioral features outperform demographic features.
- Players with lower progression and shorter sessions are more likely to churn.

---

## 💼 Business Impact

This system enables gaming platforms to:

- Identify high-risk players early  
- Launch targeted retention campaigns  
- Personalize in-game rewards  
- Reduce potential revenue loss  

Instead of reacting to churn, companies can act proactively using predictive analytics.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-player-churn-prediction.git
cd intelligent-player-churn-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook

Open:

```
notebooks/PlayerChurnPrediction.ipynb
```

### 4️⃣ (Optional) Run Streamlit App

```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## 📁 Project Structure

```
playerchurndataPrediction/
│
├── app.py                      # Streamlit or main application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── notebooks/                  # Jupyter notebooks (EDA, experiments)
│   └── PlayerChurnPrediction.ipynb
│
├── data/                       # Data storage (not tracked in git by default)
│   ├── raw/                    # Original, immutable data dumps
│   └── processed/              # Cleaned / feature-engineered datasets
│
└── assets/                     # Models and visual assets
	├── models/                 # Saved model artifacts (e.g., churn_model.pkl)
	└── figures/                # Plots and figures
```

---

## 🧠 Future Improvements

- Real-time production deployment  
- Advanced ensemble models (XGBoost, LightGBM)  
- Model monitoring & drift detection  
- SHAP-based feature explainability  
- AI-driven retention recommendation engine  

---

## 👥 Team

- Pushkar  
- Parrv  
- Malhar  
- Akshat  

---

## ⭐ Project Highlights

- End-to-end ML pipeline  
- Strong predictive performance (ROC-AUC ~0.94)  
- Business-focused application  
- Scalable for production deployment  

---

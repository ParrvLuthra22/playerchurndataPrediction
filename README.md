🎮 Intelligent Player Churn Prediction Using Machine Learning
📌 Overview
Player retention is critical in the online gaming industry. When players stop engaging with a platform, it directly impacts revenue and long‑term growth.

This project builds a machine learning–based churn prediction system that identifies players at risk of leaving based on behavioral engagement data.

The system uses supervised learning models to predict churn before it happens, enabling proactive retention strategies.

🎯 Problem Statement
Churn is defined as players exhibiting low engagement levels, indicating a high likelihood of discontinuing gameplay.

We transformed churn detection into a binary classification problem:

Low Engagement → Churn = 1

Medium/High Engagement → Churn = 0

📊 Dataset Description
40,034 player records

13 demographic and behavioral features

Key Features:
Age

Gender

Location

GameGenre

PlayTimeHours

SessionsPerWeek

AvgSessionDurationMinutes

PlayerLevel

AchievementsUnlocked

Behavioral engagement features were found to be significantly more predictive than demographic features.

⚙️ Methodology
🔹 Data Preprocessing
Removed PlayerID (identifier column)

Converted EngagementLevel to binary churn label

Applied One-Hot Encoding to categorical features

Scaled numerical features using StandardScaler

Performed 80–20 stratified train-test split

🔹 Feature Engineering
Created:

EngagementScore = SessionsPerWeek × AvgSessionDurationMinutes
This captures overall player engagement intensity.

🤖 Models Implemented
1️⃣ Logistic Regression (Baseline Model)
Accuracy: ~87%

ROC-AUC: ~0.90

2️⃣ Random Forest Classifier (Best Performing Model)
Accuracy: ~92%

ROC-AUC: ~0.94

Random Forest outperformed Logistic Regression due to its ability to capture nonlinear behavioral patterns.

📈 Evaluation Metrics
Models were evaluated using:

Accuracy

ROC-AUC Score

Precision & Recall

Confusion Matrix

A ROC-AUC above 0.90 indicates strong separation between churned and retained players.

🔍 Key Insights
SessionsPerWeek is the strongest churn predictor.

Engagement duration strongly influences retention.

Behavioral features outperform demographic attributes.

Players with lower progression and shorter sessions are more likely to churn.

💼 Business Impact
This system enables gaming companies to:

Identify high-risk players early

Launch targeted retention campaigns

Personalize in-game rewards

Reduce revenue loss

Instead of reacting to churn, companies can act proactively.

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/intelligent-player-churn-prediction.git
cd intelligent-player-churn-prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Notebook
Open:

PlayerChurnData.ipynb
Or launch the Streamlit app (if included):

streamlit run app.py
🛠 Tech Stack
Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

Streamlit

📌 Project Structure
intelligent-player-churn-prediction/
│
├── PlayerChurnData.ipynb
├── requirements.txt
├── README.md
└── app.py (optional Streamlit app)
🧠 Future Work
Real-time deployment

Integration with AI-driven retention recommendation system

Advanced ensemble optimization

Model monitoring pipeline

👥 Team Members
Pushkar

Parrv

Malhar

Akshat


import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Intelligent Player Churn Prediction",
    layout="wide"
)

st.title("🎮 Intelligent Player Churn Prediction")
st.markdown("Upload a CSV dataset to predict player churn risk.")

# Load trained model (Pipeline with preprocessing included)
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Player Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # If EngagementLevel exists, create Churn column (same as notebook)
        if "EngagementLevel" in df.columns:
            df["Churn"] = df["EngagementLevel"].apply(lambda x: 1 if x == "Low" else 0)
            df = df.drop(["PlayerID", "EngagementLevel"], axis=1)

        # Predict
        predictions = model.predict(df)
        df["Predicted_Churn"] = predictions

        st.success("✅ Prediction Completed Successfully!")

        st.subheader("🔍 Prediction Results")
        st.dataframe(df.head())

        # Show churn rate metric
        churn_rate = df["Predicted_Churn"].mean() * 100
        st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

    except Exception as e:
        st.error("⚠️ Error processing file. Please upload the correct dataset format.")

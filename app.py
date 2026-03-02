"""Streamlit app for intelligent player churn prediction.

This module exposes a small set of functions to:
* Configure the Streamlit page.
* Lazily load the trained churn prediction model.
* Process uploaded CSV files into a suitable format.
* Render predictions and basic metrics in the UI.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base directory of the project (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

# Default model path (relative to project root); update here if you move models
DEFAULT_MODEL_PATH = BASE_DIR / "assets" / "models" / "churn_model.pkl"


def configure_page() -> None:
    """Configure basic Streamlit page settings and header.

    This should be called once at the beginning of the app execution.
    """

    st.set_page_config(
        page_title="Intelligent Player Churn Prediction",
        layout="wide",
    )

    st.title("🎮 Intelligent Player Churn Prediction")
    st.markdown("Upload a CSV dataset to predict player churn risk.")


@st.cache_resource
def load_model(model_path: Optional[Path] = None):
    """Load and cache the trained churn prediction model.

    Parameters
    ----------
    model_path:
        Optional path to the serialized model. If not provided, the
        default path defined by ``DEFAULT_MODEL_PATH`` is used.

    Returns
    -------
    Any
        A scikit-learn compatible model/pipeline with a ``predict`` method.
    """

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the uploaded dataframe into model-ready features.

    Steps mirror the notebook logic:
    * If an ``EngagementLevel`` column exists, derive a binary ``Churn`` label.
    * Drop identifier and intermediate label columns (``PlayerID``, ``EngagementLevel``).

    Parameters
    ----------
    df:
        Raw dataframe loaded from the uploaded CSV file.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready to be passed into the model.
    """

    processed_df = df.copy()

    if "EngagementLevel" in processed_df.columns:
        processed_df["Churn"] = processed_df["EngagementLevel"].apply(
            lambda engagement_level: 1 if engagement_level == "Low" else 0
        )

    columns_to_drop = [
        column
        for column in ["PlayerID", "EngagementLevel"]
        if column in processed_df.columns
    ]

    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop, axis=1)

    return processed_df


def run_app() -> None:
    """Main entry point for the Streamlit churn prediction app.

    Handles file upload, preview, preprocessing, prediction, and display
    of basic churn metrics.
    """

    configure_page()

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(
            "⚠️ Trained model file not found. "
            "Please ensure the model is saved at 'assets/models/churn_model.pkl'."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Player Dataset (CSV)", type=["csv"], accept_multiple_files=False
    )

    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("⚠️ Error reading CSV file. Please upload a valid CSV dataset.")
        return

    st.subheader("📊 Dataset Preview")
    st.dataframe(input_df.head())

    try:
        features_df = preprocess_input(input_df)
        predictions = model.predict(features_df)
    except Exception:
        st.error(
            "⚠️ Error during preprocessing or prediction. "
            "Please ensure the uploaded data matches the training schema."
        )
        return

    # Attach predictions to the original data for user-friendly display
    result_df = input_df.copy()
    result_df["Predicted_Churn"] = predictions

    st.success("✅ Prediction Completed Successfully!")

    st.subheader("🔍 Prediction Results (first 5 rows)")
    st.dataframe(result_df.head())

    churn_rate = float(result_df["Predicted_Churn"].mean() * 100)
    st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")


if __name__ == "__main__":
    run_app()

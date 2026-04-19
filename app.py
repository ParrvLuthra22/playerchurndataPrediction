"""Streamlit app for AI-Powered Player Churn Advisor System.

This app extends an existing churn prediction model with:
1) CSV-based churn prediction
2) RAG retrieval of retention strategies
3) LangGraph agent explanation and recommendations
"""

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATE_PATHS = [
    BASE_DIR / "model" / "churn_model.pkl",
    BASE_DIR / "assets" / "models" / "churn_model.pkl",
]
NON_FEATURE_COLUMNS = ("PlayerID", "EngagementLevel", "Churn")


def resolve_model_path() -> Path:
    """Resolve model path from supported locations."""

    for path in MODEL_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find 'churn_model.pkl' in either 'model/' or 'assets/models/'."
    )


def configure_page() -> None:
    """Configure Streamlit page settings and title block."""

    st.set_page_config(
        page_title="AI-Powered Player Churn Advisor",
        layout="wide",
    )

    st.title("🎮 AI-Powered Player Churn Advisor")
    st.markdown(
        "Upload a CSV dataset to predict churn and generate AI-powered retention advice."
    )


@st.cache_resource
def load_model(model_path: Optional[Path] = None):
    """Load and cache the existing trained churn model (no retraining)."""

    path = Path(model_path) if model_path is not None else resolve_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def get_model_expected_features(model: Any) -> list[str]:
    """Return ordered model feature names when available."""

    return list(getattr(model, "feature_names_in_", []))


def preprocess_input(
    df: pd.DataFrame, expected_features: Optional[list[str]] = None
) -> pd.DataFrame:
    """Prepare upload dataframe for model inference.

    The saved model is assumed to include preprocessing in its pipeline.
    This function removes non-feature identifiers/labels and, when provided,
    aligns the uploaded data to model-expected feature columns.
    """

    processed_df = df.copy()

    columns_to_drop = [
        column for column in NON_FEATURE_COLUMNS if column in processed_df.columns
    ]

    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop)

    if expected_features:
        missing_columns = [
            column for column in expected_features if column not in processed_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required feature columns for prediction: "
                f"{missing_columns}. "
                f"Available columns: {list(processed_df.columns)}"
            )

        # Keep only features used by the model and preserve exact feature order.
        processed_df = processed_df[expected_features]

    return processed_df


def render_schema_summary(input_df: pd.DataFrame, expected_features: list[str]) -> None:
    """Render model schema guidance for uploaded data."""

    if not expected_features:
        st.caption("Model feature schema metadata is unavailable; using uploaded columns directly.")
        return

    missing_columns = [c for c in expected_features if c not in input_df.columns]
    extra_columns = [c for c in input_df.columns if c not in expected_features]

    with st.expander("Model Schema Check"):
        st.write(f"Expected feature columns: **{len(expected_features)}**")
        st.code(", ".join(expected_features), language="text")

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            st.success("No missing required columns.")

        if extra_columns:
            st.info(f"Extra columns detected (will be ignored if non-feature): {extra_columns}")


def run_app() -> None:
    """Run the Streamlit churn advisor app."""

    configure_page()

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(
            "⚠️ Trained model file not found. "
            "Please ensure the model exists at 'model/churn_model.pkl' or "
            "'assets/models/churn_model.pkl'."
        )
        st.caption(str(error))
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

    expected_features = get_model_expected_features(model)
    render_schema_summary(input_df, expected_features)

    try:
        features_df = preprocess_input(
            input_df,
            expected_features=expected_features if expected_features else None,
        )
        predictions = model.predict(features_df)
    except Exception as error:
        st.error(
            "⚠️ Error during preprocessing or prediction. "
            "Please ensure the uploaded data matches the training schema."
        )
        st.caption(f"Prediction error: {error}")
        return

    # Attach predictions to the original data for user-friendly display.
    result_df = input_df.copy()
    result_df["Predicted_Churn"] = predictions
    result_df["Churn_Risk_Label"] = result_df["Predicted_Churn"].map(
        lambda value: "High Churn Risk" if int(value) == 1 else "Low Churn Risk"
    )

    st.success("✅ Prediction Completed Successfully!")

    st.subheader("🔍 Prediction Results (first 5 rows)")
    st.dataframe(result_df.head())

    churn_rate = float(result_df["Predicted_Churn"].mean() * 100)
    st.metric("Predicted Churn Rate", f"{churn_rate:.2f}%")

    st.divider()
    st.subheader("🧠 AI Churn Advisor (GenAI + RAG + LangGraph)")
    st.caption(
        "The advisor analyzes one sample player row, retrieves retention strategies "
        "from the knowledge base, and generates recommendations."
    )

    high_risk_rows = result_df.index[result_df["Predicted_Churn"] == 1].tolist()
    sample_index = high_risk_rows[0] if high_risk_rows else int(result_df.index[0])

    st.write(f"Using sample row index: **{sample_index}**")
    sample_row = result_df.loc[sample_index].to_dict()
    prediction_label = str(sample_row.get("Churn_Risk_Label", "Unknown Risk"))

    try:
        with st.spinner("Running AI advisor workflow..."):
            from agent.graph import run_churn_advisor

            agent_output = run_churn_advisor(
                player_data=sample_row,
                prediction_label=prediction_label,
            )

        st.markdown(f"### Prediction: {agent_output['prediction']}")
        st.markdown("#### Reason:")
        st.write(agent_output["reason"])

        st.markdown("#### Recommendation:")
        for item in agent_output["recommendations"]:
            st.markdown(f"- {item}")

        with st.expander("Retrieved Knowledge Chunks"):
            docs = agent_output.get("retrieved_docs", [])
            if not docs:
                st.info("No retrieved documents available.")
            for idx, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {idx}**: {doc}")

    except Exception as error:
        st.warning(
            "AI advisor could not run. Predictions are still available. "
            "Check Groq API configuration and dependencies."
        )
        st.caption(f"Advisor error: {error}")


if __name__ == "__main__":
    run_app()

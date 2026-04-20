from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import streamlit as st


APP_TITLE = "📊 Invoice Intelligent System"
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class ModelPaths:
    freight_model: Path = MODELS_DIR / "predict_freight_model.pkl"
    flag_model: Path = MODELS_DIR / "predict_flag_invoice.pkl"
    flag_scaler: Path = MODELS_DIR / "scaler.pkl"


FLAG_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "freight_invoiced",
    "total_item_quantity",
    "total_item_dollars",
]


# -----------------------------
# Utility Functions
# -----------------------------
def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing required file: {path.as_posix()}")


@st.cache_resource
def load_joblib(path: Path):
    _require_file(path)
    return joblib.load(path)


def predict_freight(dollars: float, paths: ModelPaths) -> float:
    model = load_joblib(paths.freight_model)
    df = pd.DataFrame({"Dollars": [dollars]})
    return float(model.predict(df)[0])


def predict_flag(features: dict, paths: ModelPaths) -> int:
    model = load_joblib(paths.flag_model)
    scaler = load_joblib(paths.flag_scaler)

    df = pd.DataFrame([features], columns=FLAG_FEATURES)
    x_scaled = scaler.transform(df)

    return int(model.predict(x_scaled)[0])


# -----------------------------
# UI Functions
# -----------------------------
def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>📊 Invoice Intelligent System</h1>
        <p style='text-align: center;'>AI-powered Freight Prediction & Invoice Risk Detection</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()


def render_sidebar() -> Literal["Freight cost", "Invoice flag"]:
    st.sidebar.title("📌 Navigation")

    choice = st.sidebar.radio(
        "Choose Module",
        ["🚚 Freight cost", "⚠️ Invoice flag"],
    )

    st.sidebar.info("💡 Tip: Use realistic values for better predictions")

    return "Freight cost" if "Freight" in choice else "Invoice flag"


# -----------------------------
# Freight Page
# -----------------------------
def render_freight_page(paths: ModelPaths) -> None:
    st.subheader("🚚 Freight Cost Prediction")

    st.info("Enter invoice amount to predict shipping (freight) cost")

    dollars = st.slider("💰 Invoice Dollars", 0, 10000, 1500)

    if st.button("🔍 Predict Freight"):
        try:
            pred = predict_freight(dollars, paths)

            st.success("✅ Prediction Complete!")

            col1, col2 = st.columns(2)
            col1.metric("💰 Invoice Amount", f"{dollars}")
            col2.metric("🚚 Predicted Freight", f"{pred:.2f}")

        except Exception as e:
            st.error(str(e))


# -----------------------------
# Flag Page
# -----------------------------
def render_flag_page(paths: ModelPaths) -> None:
    st.subheader("⚠️ Invoice Risk Detection")

    st.warning("Detect suspicious invoices using ML model")

    col1, col2 = st.columns(2)

    with col1:
        invoice_quantity = st.number_input("📦 Invoice Quantity", 0.0, 1000.0, 10.0)
        invoice_dollars = st.number_input("💰 Invoice Dollars", 0.0, 100000.0, 1500.0)
        freight_invoiced = st.number_input("🚚 Freight Invoiced", 0.0, 10000.0, 25.0)

    with col2:
        total_item_quantity = st.number_input("📊 Total Item Quantity", 0.0, 1000.0, 10.0)
        total_item_dollars = st.number_input("💵 Total Item Dollars", 0.0, 100000.0, 1498.0)

    if st.button("🔍 Predict Risk"):
        features = {
            "invoice_quantity": float(invoice_quantity),
            "invoice_dollars": float(invoice_dollars),
            "freight_invoiced": float(freight_invoiced),
            "total_item_quantity": float(total_item_quantity),
            "total_item_dollars": float(total_item_dollars),
        }

        try:
            pred = predict_flag(features, paths)

            if pred == 1:
                st.error("🚨 FLAGGED: Suspicious Invoice Detected!")
            else:
                st.success("✅ SAFE: Invoice looks normal")

            st.subheader("📋 Input Summary")
            st.dataframe(pd.DataFrame([features]), use_container_width=True)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# Main App
# -----------------------------
def main() -> None:
    render_header()
    page = render_sidebar()
    paths = ModelPaths()

    if page == "Freight cost":
        render_freight_page(paths)
    else:
        render_flag_page(paths)


if __name__ == "__main__":
    main()
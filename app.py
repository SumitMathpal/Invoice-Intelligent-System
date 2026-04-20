from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import streamlit as st
import re

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO


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


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        # First try: embedded/selectable text
        selectable_text = "\n".join(page.get_text() for page in doc).strip()
        if selectable_text:
            return selectable_text

        # Fallback: OCR for scanned PDFs (image-only)
        ocr_chunks: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=220)
            img_bytes = pix.tobytes("png")
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            try:
                ocr_chunks.append(pytesseract.image_to_string(img))
            except pytesseract.TesseractNotFoundError as e:
                raise RuntimeError(
                    "Tesseract OCR not found. Install it and ensure `tesseract.exe` is on PATH, "
                    "or set `pytesseract.pytesseract.tesseract_cmd` to the full path."
                ) from e
        return "\n".join(ocr_chunks).strip()
    finally:
        doc.close()


_CURRENCY_RE = re.compile(r"[^\d.\-]+")


def _to_number(raw: str) -> float:
    cleaned = raw.replace(",", "").strip()
    cleaned = _CURRENCY_RE.sub("", cleaned)
    return float(cleaned)


def parse_flag_features_from_text(text: str) -> tuple[dict[str, float], list[str]]:
    """
    Best-effort extraction for the 5 model features from selectable PDF text.
    Returns: (features_found, missing_feature_names)
    """
    t = " ".join(text.split())
    patterns: dict[str, list[re.Pattern[str]]] = {
        "invoice_quantity": [
            re.compile(r"(?:invoice\s*quantity|invoice\s*qty)\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.I),
            re.compile(r"\bqty\b\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.I),
            re.compile(r"\bquantity\b\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.I),
        ],
        "invoice_dollars": [
            re.compile(r"(?:invoice\s*amount|invoice\s*dollars)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
            re.compile(r"(?:grand\s*total|total\s*amount|amount\s*due)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        ],
        "freight_invoiced": [
            re.compile(r"(?:freight\s*invoiced|freight\s*charges?)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
            re.compile(r"(?:shipping|delivery)\s*(?:charges?)?\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        ],
        "total_item_quantity": [
            re.compile(r"(?:total\s*item\s*quantity|total\s*items?\s*qty)\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.I),
            re.compile(r"(?:items?\s*total\s*qty)\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.I),
        ],
        "total_item_dollars": [
            re.compile(r"(?:total\s*item\s*dollars|total\s*items?\s*amount)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
            re.compile(r"(?:items?\s*total)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I),
        ],
    }

    features: dict[str, float] = {}
    for key in FLAG_FEATURES:
        for pat in patterns.get(key, []):
            m = pat.search(t)
            if not m:
                continue
            try:
                features[key] = _to_number(m.group(1))
                break
            except Exception:
                continue

    missing = [k for k in FLAG_FEATURES if k not in features]
    return features, missing


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

    mode = st.radio(
        "Choose input method",
        ["Manual input", "Upload PDFs (batch check)", "Upload Excel/CSV (batch check)"],
        horizontal=True,
    )

    if mode == "Upload Excel/CSV (batch check)":
        st.info(
            "Upload an Excel/CSV file where each row is one invoice. "
            "Required columns are the 5 model features."
        )

        with st.expander("✅ Required columns", expanded=True):
            st.code("\n".join(FLAG_FEATURES))

        sample_df = pd.DataFrame(
            [
                {
                    "invoice_quantity": 10,
                    "invoice_dollars": 1500,
                    "freight_invoiced": 25,
                    "total_item_quantity": 10,
                    "total_item_dollars": 1498,
                },
                {
                    "invoice_quantity": 10,
                    "invoice_dollars": 1500,
                    "freight_invoiced": 800,
                    "total_item_quantity": 10,
                    "total_item_dollars": 1498,
                },
            ]
        )
        st.download_button(
            "⬇️ Download sample CSV template",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="invoice_flag_template.csv",
            mime="text/csv",
        )

        up = st.file_uploader(
            "➕ Add Excel/CSV file",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=False,
        )

        if up is None:
            return

        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        st.subheader("📥 Uploaded data preview")
        st.dataframe(df.head(50), use_container_width=True)

        missing_cols = [c for c in FLAG_FEATURES if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return

        run = st.button("✅ Check all rows")
        if not run:
            return

        work = df.copy()
        for c in FLAG_FEATURES:
            work[c] = pd.to_numeric(work[c], errors="coerce")

        bad_rows = work[FLAG_FEATURES].isna().any(axis=1)
        results: list[dict] = []

        for idx, row in work.iterrows():
            if bad_rows.loc[idx]:
                results.append(
                    {
                        "row": int(idx),
                        "decision": "Not checked",
                        "predicted_flag": None,
                        "status": "needs_review",
                        "error": "missing_or_non_numeric_values",
                    }
                )
                continue

            features = {k: float(row[k]) for k in FLAG_FEATURES}
            try:
                pred = predict_flag(features, paths)
                decision = "Not correct (FLAGGED)" if pred == 1 else "Correct (SAFE)"
                results.append(
                    {
                        "row": int(idx),
                        "decision": decision,
                        "predicted_flag": int(pred),
                        "status": "checked",
                        "error": "",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "row": int(idx),
                        "decision": "Not checked",
                        "predicted_flag": None,
                        "status": "error",
                        "error": str(e),
                    }
                )

        out = pd.DataFrame(results)
        st.subheader("📄 Batch Results")
        st.dataframe(out, use_container_width=True)

        checked = out[out["status"] == "checked"]
        if not checked.empty:
            st.subheader("📊 Summary")
            safe = int((checked["predicted_flag"] == 0).sum())
            flagged = int((checked["predicted_flag"] == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Checked rows", f"{len(checked)}")
            c2.metric("SAFE", f"{safe}")
            c3.metric("FLAGGED", f"{flagged}")

        return

    if mode == "Upload PDFs (batch check)":
        st.info(
            "Upload one or more invoice PDFs and check them together. "
            "Works best for PDFs with selectable text (not scanned images)."
        )

        uploaded = st.file_uploader(
            "➕ Add PDF invoice(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        check = st.button("✅ Check all invoices", disabled=not uploaded)
        if check:
            rows: list[dict] = []

            for f in uploaded or []:
                try:
                    text = extract_text_from_pdf_bytes(f.getvalue())
                except Exception as e:
                    rows.append(
                        {
                            "file": f.name,
                            "decision": "Not checked",
                            "predicted_flag": None,
                            "status": "error",
                            "missing_fields": "",
                            "extracted_text_chars": None,
                            "text_preview": "",
                            "error": str(e),
                        }
                    )
                    continue

                text_stripped = text.strip()
                extracted_chars = len(text_stripped)
                preview = text_stripped[:220].replace("\n", " ") if text_stripped else ""

                if extracted_chars == 0:
                    rows.append(
                        {
                            "file": f.name,
                            "decision": "Not checked",
                            "predicted_flag": None,
                            "status": "scanned_or_no_text",
                            "missing_fields": ", ".join(FLAG_FEATURES),
                            "extracted_text_chars": extracted_chars,
                            "text_preview": "",
                            "error": "",
                        }
                    )
                    continue

                features, missing = parse_flag_features_from_text(text)
                if missing:
                    rows.append(
                        {
                            "file": f.name,
                            "decision": "Not checked",
                            "predicted_flag": None,
                            "status": "needs_review",
                            "missing_fields": ", ".join(missing),
                            "extracted_text_chars": extracted_chars,
                            "text_preview": preview,
                            "error": "",
                        }
                    )
                    continue

                try:
                    pred = predict_flag(features, paths)
                    decision = "Not correct (FLAGGED)" if pred == 1 else "Correct (SAFE)"
                    rows.append(
                        {
                            "file": f.name,
                            "decision": decision,
                            "predicted_flag": pred,
                            "status": "checked",
                            "missing_fields": "",
                            "extracted_text_chars": extracted_chars,
                            "text_preview": preview,
                            "error": "",
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "file": f.name,
                            "decision": "Not checked",
                            "predicted_flag": None,
                            "status": "error",
                            "missing_fields": "",
                            "extracted_text_chars": extracted_chars,
                            "text_preview": preview,
                            "error": str(e),
                        }
                    )

            st.subheader("📄 Batch Results")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if any(r["status"] == "needs_review" for r in rows):
                st.warning(
                    "Some PDFs are missing required numbers (see `missing_fields`). "
                    "If the PDF is scanned, convert it to selectable text (OCR) or use Manual input."
                )
            if any(r["status"] == "scanned_or_no_text" for r in rows):
                st.warning(
                    "Some PDFs have **no selectable text** (`status=scanned_or_no_text`). "
                    "This feature currently reads text from the PDF; scanned images will return 0 characters."
                )
        return

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
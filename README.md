# 📊 Invoice Intelligent System

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python Version](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Invoice Intelligent System** is an AI-powered analytics and risk management pipeline. Built with **Streamlit** and **Scikit-Learn**, it enables business operations and audit teams to automate freight cost estimation and detect anomalous or fraudulent invoices instantly.

---

## 🗺️ Interactive Navigation Map

Select a section below to jump directly to it:

*   [🚀 Quick Start](#-quick-start)
*   [🛠️ Core Features](#%EF%B8%8F-core-features)
*   [📊 Data Schema & Inputs](#-data-schema--inputs)
*   [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
*   [⚙️ Requirements & Dependencies](#%EF%B8%8F-requirements--dependencies)
*   [⚡ Standalone Inference & CLI](#-standalone-inference--cli)
*   [🔍 Troubleshooting FAQs](#-troubleshooting-faqs)

---

## 🛠️ Core Features

Below is a detailed breakdown of the system capabilities. Click to expand each panel:

<details>
<summary><b>⚠️ Invoice Risk Detection & Flagging</b></summary>

Detect suspicious or fraudulent invoices using a pre-trained classification model. The system flags discrepancies between invoice details, items, quantities, and dollars.

**Three interactive input modes are supported:**
1.  **Manual Input UI**: Input values via sliders and numeric fields. Get instant classification output (`SAFE` or `RISK FLAGGED`).
2.  **Excel & CSV Batch Check**: Upload a structured spreadsheet. The system processes the contents, runs validation checks, and returns a download-ready result table.
3.  **PDF Batch Processing**: Upload PDF invoices. The system will read the text structure using **PyMuPDF**. If the PDF is scanned (image-only), it automatically falls back to **Tesseract OCR** to extract features via dynamic regular expression heuristics.
</details>

<details>
<summary><b>🚚 Freight Cost Prediction</b></summary>

Estimate shipping (freight) costs based on the total dollar amount of the invoice using an ML regression model. Useful for logistics planning and auditing carrier invoices for overcharging.
</details>

---

## 🏗️ System Architecture

The following diagram illustrates how raw invoice data flows through the preprocessing and parsing libraries to feed the machine learning models:

```ascii
[ User Uploads PDF/CSV or Enters Data ]
                  │
                  ▼
   ┌──────────────────────────────┐
   │    Data Extraction Layer     │
   │                              │
   │  - PyMuPDF / Fitz (Text)     │
   │  - Tesseract (Scanned OCR)   │
   │  - Pandas (CSV parsing)      │
   └──────────────┬───────────────┘
                  │ (Raw Text/DataFrame)
                  ▼
   ┌──────────────────────────────┐
   │    Feature Parser & Regex    │
   │                              │
   │  - Extract 5 Numerical Cols  │
   │  - Sanitize Numeric Formats  │
   └──────────────┬───────────────┘
                  │ (Clean Features Dict)
                  ▼
   ┌──────────────────────────────┐
   │     Pre-processing Scaler    │
   │                              │
   │  - scaler.pkl                │
   └──────────────┬───────────────┘
                  │ (Scaled Numpy Array)
                  ▼
   ┌──────────────────────────────┐
   │     Machine Learning Models  │
   │                              │
   │  - predict_flag_invoice.pkl  │
   │  - predict_freight_model.pkl │
   └──────────────┬───────────────┘
                  │
                  ▼
     [ Streamlit Results Display ]
     - Decision: SAFE vs FLAGGED
     - Metrics / Batch Dataframes
```

---

## 🚀 Quick Start

Get your local instance up and running in a few simple steps.

### 1. Setup Virtual Environment
Clone the repository, navigate to the directory, and run:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```

### 4. Direct CLI Smoke Test
Test the python models instantly from the terminal without launching the UI:
```bash
python smoke_test.py
```

---

## 📊 Data Schema & Inputs

To perform risk detection, the model requires exactly **5 numerical features**. If uploading a spreadsheet (CSV/Excel) or auditing PDFs, ensure these fields are present and properly parsed:

| Feature Name | Description | Example Value |
| :--- | :--- | :--- |
| `invoice_quantity` | Total item count listed on the invoice wrapper | `10.0` |
| `invoice_dollars` | Grand total dollar amount of the invoice | `1500.0` |
| `freight_invoiced` | Shipping and transport charges invoiced | `25.0` |
| `total_item_quantity` | Sum of individual line-item quantities | `10.0` |
| `total_item_dollars` | Sum of individual line-item dollar values | `1498.0` |

<details>
<summary><b>💡 Interactive Data Rules & Insights</b></summary>

*   **Risk Flags:** An invoice is flagged as suspicious when variables like `total_item_dollars` diverge drastically from `invoice_dollars` (excluding freight costs), or if `freight_invoiced` is phenomenally high relative to the raw quantities.
*   **Template Download:** Launch the Streamlit app and select the **Upload Excel/CSV** option to download a pre-formatted template with sample rows.
</details>

---

## ⚙️ Requirements & Dependencies

The primary dependencies are defined in `requirements.txt`:
*   `streamlit` - Web application interface
*   `pandas` & `numpy` - Data wrangling and transformation
*   `scikit-learn` & `joblib` - Machine learning models & load routines
*   `pymupdf` (fitz) - High performance PDF text parsing
*   `pytesseract` & `Pillow` - Optical Character Recognition (OCR) for images
*   `openpyxl` - Excel file reader support

<details>
<summary><b>⚠️ Action Required: Setting up Tesseract OCR (Optional)</b></summary>

To parse scanned/image-based PDFs, the system requires the Tesseract engine installed on your computer.

*   **Windows**:
    1.  Download installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
    2.  Add the installation path (`C:\Program Files\Tesseract-OCR`) to your system PATH environment variable, or configure it in Python.
*   **macOS**:
    ```bash
    brew install tesseract
    ```
*   **Linux (Ubuntu/Debian)**:
    ```bash
    sudo apt-get install tesseract-ocr
    ```
</details>

---

## ⚡ Standalone Inference & CLI

You can run predictions inside other applications or workflows via Python script hooks:

```python
from core.predict import ModelPaths, predict_flag, predict_freight

paths = ModelPaths()

# Predict Carriage/Shipping cost
estimated_freight = predict_freight(1500.0, paths)
print(f"Predicted Freight: ${estimated_freight:.2f}")

# Check invoice safety 
invoice_data = {
    "invoice_quantity": 10.0,
    "invoice_dollars": 1500.0,
    "freight_invoiced": 25.0,
    "total_item_quantity": 10.0,
    "total_item_dollars": 1498.0
}
flag = predict_flag(invoice_data, paths)
print("Safety Status:", "⚠️ FLAGGED" if flag == 1 else "✅ SAFE")
```

---

## 🔍 Troubleshooting FAQs

Here are quick solutions to common errors.

<details>
<summary><b>Q1: Missing required file error on startup?</b></summary>

Ensure the directory structure has the saved models within the `models/` directory:
- `models/predict_flag_invoice.pkl`
- `models/predict_freight_model.pkl`
- `models/scaler.pkl`
</details>

<details>
<summary><b>Q2: TesseractNotFoundError: Tesseract OCR not found?</b></summary>

This means the application is attempting to OCR a scanned invoice but can't find the system binary.
1.  Make sure Tesseract is installed.
2.  Add its folder path to your user/system environment variables.
3.  Restart your terminal/IDE before running the Streamlit app.
</details>

<details>
<summary><b>Q3: Uploaded CSV/Excel returns columns error?</b></summary>

Verify that the column headers in your spreadsheet exactly match the feature list:
`invoice_quantity`, `invoice_dollars`, `freight_invoiced`, `total_item_quantity`, `total_item_dollars`
(case-sensitive, lowercase, underscores).
</details>

---

## 📄 License & Contact

This project is licensed under the MIT License. Developed for intelligent accounting automation.

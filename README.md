## Invoice Intelligent System (Streamlit)

This repo contains a simple Streamlit UI to run:
- **Freight cost prediction** using `models/predict_freight_model.pkl`
- **Invoice flagging** using `models/predict_flag_invoice.pkl` + `models/scaler.pkl`

### Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Quick smoke test (no UI)

```bash
python smoke_test.py
```

### Project structure

- `app.py`: Streamlit app entrypoint
- `models/`: saved models and scaler
- `inference/`: standalone prediction scripts
- `Freight_cost_prediction/`, `invoice_flagging/`: training code


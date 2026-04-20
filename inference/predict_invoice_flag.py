import joblib
import pandas as pd 

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_model(model_path:str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path,"rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

     Parameters
     -----------
     input_data : dict

     Returns
     -------
     pd.DataFrame with predict flag
    """
    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    input_df["Predicted_flag"] = model.predict(input_scaled).round().astype(int)
    return input_df
if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10],
        "invoice_dollars": [1500],
        "freight_invoiced": [25],
        "total_item_quantity": [10],
        "total_item_dollars": [1498],
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)
import os
import joblib
from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modeling_evaluating import train_random_forest, evaluate_model

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "freight_invoiced",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    df = load_invoice_data()
    df = apply_labels(df)

    print(df.columns)  # Debug check

    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, "models/scaler.pkl")

    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_model(grid_search.best_estimator_, X_test_scaled, y_test, "Random Forest Classifier")

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, "models/predict_flag_invoice.pkl")


if __name__ == "__main__":
    main()
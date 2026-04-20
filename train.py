from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modeling_evaluation import train_random_forest, evaluate_model


FEATURES = ["invoice_quantity","invoice_dollars","Freight_invoiced","total_item_quantity","total_item_dollars"]

TARGET = "flag_invoice"

def main():
    df = laod_invoice_data()
    df  = apply_labels(df)

    X_train, X_test,y_train = split_data(df,FEATURES,)
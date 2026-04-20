import joblib
from pathlib import Path
from data_processing import load_vendor_invoice_data, prepare_features, split_data
from model_evalution import (train_linear_regression,train_decision_tree,train_random_forest,evalute_model)

def main():
    # Construct absolute paths based on the script's location
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    candidates = [
        data_dir / "inventory.db",
        data_dir / ".db",
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    
    if db_path is None:
        raise FileNotFoundError(
            "Database file not found. Expected one of:\n"
            + "\n".join(f"- {p.as_posix()}" for p in candidates)
        )
        
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    #load data 
    df=load_vendor_invoice_data(str(db_path))
    #prepare data
    X,y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X,y)
    #Train data
    lr_model = train_linear_regression(X_train,y_train)
    dt_model = train_decision_tree(X_train,y_train)
    rf_model = train_random_forest(X_train,y_train)
    #Evalute models 
    results=[]
    results.append(evalute_model(lr_model,X_test,y_test,"Linear Regression"))
    results.append(evalute_model(dt_model,X_test,y_test,"Decision tree"))
    results.append(evalute_model(rf_model,X_test,y_test,"Random Forest Regression"))
    #Select Best model(lowres MSE)
    best_model_info = min(results,key=lambda x:x["mae"])
    best_model_name = best_model_info["model_name"]
    best_model ={"Linear Regression": lr_model,"Decision tree":dt_model,"Random Forest Regression":rf_model}[best_model_name]
    #Save best model
    model_path = model_dir/"predict_freight_model.pkl"
    joblib.dump(best_model,model_path)
    print(f"\nBest model saved:{best_model_name}")
    print(f"Model path:{model_path}")
if __name__ == "__main__":
    main()
    
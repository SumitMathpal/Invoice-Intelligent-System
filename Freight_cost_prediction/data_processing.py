import pandas as pd 
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(db_path: str):
    """
    Load vendor data from sql_lite Database.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query,conn)
    return df

def prepare_features(df: pd.DataFrame):
    """Select Features and target variable """
    X= df[["Dollars"]]
    y=df[["Freight"]]
    return X, y

def split_data(X,y, test_size=0.2, random_state=42):
    """Split data into train and test size"""
    return train_test_split(X,y,test_size=test_size,random_state=random_state)
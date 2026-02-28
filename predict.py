import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = joblib.load(os.path.join(BASE_DIR, "all_models.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

def predict(model_name, values):
    
    if model_name not in models:
        return f"Model '{model_name}' not found. Available: {list(models.keys())}"
    
    if len(values) != len(columns):
        return f"Expected {len(columns)} values but got {len(values)}"

    df = pd.DataFrame([values], columns=columns)

    if model_name in ["Logistic", "KNN"]:
        df = scaler.transform(df)

    pred = models[model_name].predict(df)[0]

    return "Yes" if pred == 1 else "No"

if __name__ == "__main__":
    result = predict("Logistic", [1018,25,20,85,70,2.5,60,18])
    print("Rain Prediction:", result)
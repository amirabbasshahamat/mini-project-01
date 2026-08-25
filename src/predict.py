#%% import libraries

import json
import joblib
import pandas as pd


#%% load model

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = 0.3

#%% prediction function

def predict_transaction(input_json):
    data = json.loads(input_json)
    input_data = pd.DataFrame([data], columns=scaler.feature_names_in_)
    input_scaled = scaler.transform(input_data)
    probability = float(model.predict_proba(input_scaled)[0, 1])
    class_id = int(probability >= threshold)

    return {
        "prediction": "Fraud" if class_id == 1 else "Not Fraud",
        "class_id": class_id,
        "probability": probability,
        "threshold": threshold,
        "status": "success"}

#%%

test_input = """
{
    "Time": 406,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}
"""
result = predict_transaction(test_input)
print(json.dumps(result, indent=1))
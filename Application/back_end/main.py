from typing import Union
import pandas as pd
from fastapi import FastAPI
from pydantic import *
import joblib



app = FastAPI()
class Payload(BaseModel):
    Pregnancies: int
    Age: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float

predicts=joblib.load('../../Model/disease_prediction_model_version_two.pkl')
diabetes_scaler = joblib.load('../../Model/scaler.pkl')    

@app.get("/")
def read_root():
    return {"Name": "Akash Sureshkumar",
            "Subject id":"503",
            "Project Name": "Disease Prediction System"
            }





@app.post("/predict")
def predictss(payload: Payload):
    print(payload)
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    scaled_data = diabetes_scaler.transform(df)
    y_hat = predicts.predict(scaled_data)
    return {"prediction": int(y_hat[0])}


# import json
# import requests
# test_case_5 = {
#     "Pregnancies": 8,
#     "Age": 50,
#     "Glucose": 180,
#     "BloodPressure": 90,
#     "SkinThickness": 45,
#     "Insulin": 230,
#     "BMI": 40.0,
#     "DiabetesPedigreeFunction": 0.750
# }

# testingInputData= json.dumps(test_case_5, indent=4)
# r=predict(testingInputData)
# print(r)





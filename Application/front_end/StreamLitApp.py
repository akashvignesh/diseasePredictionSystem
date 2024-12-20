import math
from collections import defaultdict
import pandas as pd
import streamlit as st
import requests
TestData = pd.read_csv("../../Model/diabetes.csv") 

slider_fields = [
    'Pregnancies',
    'Age',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction']

streamlit_field_data = defaultdict(dict)
user_options={}
st.title('Disease Prediction System')
for field in slider_fields:
    streamlit_field_data["slider_fields"][field] = [math.floor(TestData[field].min()), math.ceil(TestData[field].max())]

import json
json.dump(streamlit_field_data, open("../front_end/streamlit_options.json", "w"), indent=2)
StreamLit_SlideBar= json.load(open("../front_end/streamlit_options.json"))

for field_name, range in StreamLit_SlideBar["slider_fields"].items():
    min_val, max_val = range
    current_value = round((min_val + max_val)/2)
    user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value=current_value)

if st.button('Predict'):
    data = json.dumps(user_options, indent=2)
    r = requests.post('http://localhost:8002/predict', data=data)
    st.write(user_options)
    st.write(r.json())
import math
import streamlit as st
import requests
import os
import json
# from collections import defaultdict
# import pandas as pd
# TestData = pd.read_csv("../../diabetes.csv") 

# slider_fields = [
#     'Pregnancies',
#     'Age',
#     'Glucose',
#     'BloodPressure',
#     'SkinThickness',
#     'Insulin',
#     'BMI',
#     'DiabetesPedigreeFunction']

# streamlit_field_data = defaultdict(dict)


# for field in slider_fields:
#     streamlit_field_data["slider_fields"][field] = [math.floor(TestData[field].min()), math.ceil(TestData[field].max())]

# json.dump(streamlit_field_data, open("../front_end/streamlit_options.json", "w"), indent=2)

# --- App Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title('🩺 Diabetes Prediction System')
st.write("""
Welcome to the Diabetes Prediction System! Please input your health parameters in the sidebar, and we will predict the likelihood of diabetes using our Machine Learning Models.
""")

# --- Load Configuration ---
user_options = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "streamlit_options.json")

try:
    with open(file_path, 'r') as file:
        StreamLit_SlideBar = json.load(file)
except FileNotFoundError:
    st.error("Configuration file not found. Please ensure 'streamlit_options.json' exists.")
    st.stop()

# --- Sidebar for User Input ---
st.sidebar.title("Input Parameters")
st.sidebar.write("Adjust the sliders below to input your health details:")

for field_name, range_values in StreamLit_SlideBar.get("slider_fields", {}).items():
    min_val, max_val = range_values
    default_value = round((min_val + max_val) / 2)
    user_options[field_name] = st.sidebar.slider(
        f"{field_name.replace('_', ' ').title()}",
        min_val, max_val,
        value=default_value
    )

# --- Prediction Button ---
if st.sidebar.button('🚀 Predict') or st.button('Predict'):
    st.sidebar.success('Submitting your data for prediction...')
    
    try:
        data = json.dumps(user_options, indent=2)
        response = requests.post('http://159.203.101.120:8008/predict', data=data)
        
        if response.status_code == 200:
            result = response.json()
            st.success('✅ Prediction Successful!')
            st.write("### 📊 Input Parameters:")
            st.json(user_options)
            
            st.write("### 🧠 Prediction Result:")
            st.write(f"**Prediction:** {result.get('prediction')}")
            st.write(f"**Result:** {'Have Diabetes' if result.get('prediction') == 1 else 'Not Have Diabetes'}")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection Error: {e}")

# --- Additional Insights ---
st.markdown("---")
st.write("""
#### 📚 About the Model
Our model uses advanced machine learning techniques to predict diabetes likelihood based on the health parameters you provide. 
Make sure to consult a medical professional for accurate diagnosis and advice.
""")

# --- Footer ---
st.markdown("---")
st.write("🚀 *Built with ❤️ using Streamlit & Machine Learning models.*")
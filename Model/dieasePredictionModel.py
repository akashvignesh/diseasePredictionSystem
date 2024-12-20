
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report,root_mean_squared_error
from imblearn.over_sampling import SMOTE
import joblib
import sqlite3
from sqlite3 import Error
from ydata_profiling import ProfileReport # type: ignore
import mlflow
from mlflow.models import infer_signature


# Load dataset
beforeDataClean = pd.read_csv("D:\Code\ESDS_503\GIT\diseasePredictionSystem\Model\diabetes.csv")
# Generate Profile Report
print("Generating profile report...")
profile = ProfileReport(beforeDataClean, title="Diabetes Dataset Profiling Report")
profile.to_notebook_iframe()
# Compute the correlation matrix for numerical columns
numerical_columns = beforeDataClean.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()



# Data Cleaning
afterDataClean = beforeDataClean.copy(deep=True)
afterDataClean[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = (
    afterDataClean[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN))
afterDataClean['Glucose'] = afterDataClean['Glucose'].fillna(round(afterDataClean['Glucose'].mean()))
afterDataClean['BloodPressure'] = afterDataClean['BloodPressure'].fillna(round(afterDataClean['BloodPressure'].mean()))
afterDataClean['SkinThickness'] = afterDataClean['SkinThickness'].fillna(round(afterDataClean['SkinThickness'].median()))
afterDataClean['Insulin'] = afterDataClean['Insulin'].fillna(round(afterDataClean['Insulin'].median()))
afterDataClean['BMI'] = afterDataClean['BMI'].fillna(round(afterDataClean['BMI'].median()))

# Check for remaining NaN values
print("Remaining NaN values:", afterDataClean.isnull().sum())


# Final data for processing
data = afterDataClean


# Database Connection Functions
def create_connection(db_file, delete_db=False):
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

# Create database and tables
dbConnection = create_connection("normalizedData.db", True)

patientTable = '''CREATE TABLE Patient (
    PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
    Pregnancies INTEGER,
    Age INT
);'''
healthMatrixTable = '''CREATE TABLE HealthMetrics (
    PatientID INTEGER,
    Glucose INTEGER,
    BloodPressure INTEGER,
    SkinThickness INTEGER,
    Insulin INTEGER,
    BMI FLOAT,
    DiabetesPedigreeFunction FLOAT,
    FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
);'''
diseaseOutcomeTable = '''CREATE TABLE Outcome (
    PatientID INTEGER,
    DiabetesStatus INTEGER,
    FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
);'''

create_table(dbConnection, patientTable)
create_table(dbConnection, healthMatrixTable)
create_table(dbConnection, diseaseOutcomeTable)

# Insert data into tables
def insert_patient_data(conn, data):
    cursor = conn.cursor()
    for _, row in data.iterrows():
        cursor.execute(
            "INSERT INTO Patient (Pregnancies, Age) VALUES (?, ?)",
            (row["Pregnancies"], row["Age"])
        )
    conn.commit()

def insert_health_metrics(conn, data):
    cursor = conn.cursor()
    patient_ids = cursor.execute("SELECT PatientID FROM Patient").fetchall()
    for _, (row, patient_id) in enumerate(zip(data.iterrows(), patient_ids)):
        row = row[1]
        cursor.execute(
            """
            INSERT INTO HealthMetrics (
                PatientID, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id[0],
                row["Glucose"],
                row["BloodPressure"],
                row["SkinThickness"],
                row["Insulin"],
                row["BMI"],
                row["DiabetesPedigreeFunction"]
            )
        )
    conn.commit()

def insert_outcome(conn, data):
    cursor = conn.cursor()
    patient_ids = cursor.execute("SELECT PatientID FROM Patient").fetchall()
    for _, (row, patient_id) in enumerate(zip(data.iterrows(), patient_ids)):
        row = row[1]
        cursor.execute(
            "INSERT INTO Outcome (PatientID, DiabetesStatus) VALUES (?, ?)",
            (patient_id[0], row["Outcome"])
        )
    conn.commit()

insert_patient_data(dbConnection, data)
insert_health_metrics(dbConnection, data)
insert_outcome(dbConnection, data)
print("Data inserted successfully!")

# Load data from database
dataFromTable = '''SELECT 
    p.PatientID, 
    p.Pregnancies, 
    p.Age, 
    hm.Glucose, 
    hm.BloodPressure, 
    hm.SkinThickness, 
    hm.Insulin, 
    hm.BMI, 
    hm.DiabetesPedigreeFunction, 
    o.DiabetesStatus AS Outcome
FROM Patient p
JOIN HealthMetrics hm ON p.PatientID = hm.PatientID
JOIN Outcome o ON p.PatientID = o.PatientID;'''
data = pd.read_sql_query(dataFromTable, dbConnection)
print(data.head())
dbConnection.close()
print(data.describe())


# Data Preprocessing -- Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(data)


std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(data)


# Split features and target
X = data.drop(['Outcome','PatientID'], axis=1)
y = data['Outcome']


# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", np.bincount(y_res))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')


# Define models
log_reg = LogisticRegression(max_iter=200, random_state=42)

gradient_boosting = GradientBoostingClassifier(random_state=42)

# Random Forest Hyperparameter Tuning
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced',random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_



# Evaluate
models = {'Logistic Regression': log_reg, 'Random Forest': best_rf, 'Gradient Boosting': gradient_boosting}
results = {}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'Mean CV F1 Score': cv_f1_scores.mean(),
        'Std CV F1 Score': cv_f1_scores.std(),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Confusion Matrix': cm,
        'TP': cm[1, 1],
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0]
    }

results_df = pd.DataFrame(results).T
print(results_df)




# Save best model
best_model_name = results_df['Accuracy'].idxmax()
joblib.dump(models[best_model_name], 'disease_prediction_model.pkl')
print(best_model_name)

predicts=joblib.load('disease_prediction_model.pkl')
# Test case 1: Non-diabetic (Expected: 0)
test_case_1 = pd.DataFrame({
    'Pregnancies': [1],
    'Age': [22],
    'Glucose': [85],
    'BloodPressure': [70],
    'SkinThickness': [20],
    'Insulin': [50],
    'BMI': [25.5],
    'DiabetesPedigreeFunction': [0.201]
})

# Test case 2: Diabetic (Expected: 1)
test_case_2 = pd.DataFrame({
    'Pregnancies': [5],
    'Age': [45],
    'Glucose': [165],
    'BloodPressure': [85],
    'SkinThickness': [40],
    'Insulin': [150],
    'BMI': [35.0],
    'DiabetesPedigreeFunction': [0.543]
})

# Test case 3: Borderline (Uncertain outcome)
test_case_3 = pd.DataFrame({
    'Pregnancies': [3],
    'Age': [33],
    'Glucose': [110],
    'BloodPressure': [70],
    'SkinThickness': [25],
    'Insulin': [90],
    'BMI': [28.0],
    'DiabetesPedigreeFunction': [0.350]
})

# Test case 4: Non-diabetic (Expected: 0)
test_case_4 = pd.DataFrame({
    'Pregnancies': [0],
    'Age': [19],
    'Glucose': [75],
    'BloodPressure': [60],
    'SkinThickness': [18],
    'Insulin': [40],
    'BMI': [22.0],
    'DiabetesPedigreeFunction': [0.150]
})

# Test case 5: Diabetic (Expected: 1)
test_case_5 = pd.DataFrame({
    'Pregnancies': [8],
    'Age': [50],
    'Glucose': [180],
    'BloodPressure': [90],
    'SkinThickness': [45],
    'Insulin': [230],
    'BMI': [40.0],
    'DiabetesPedigreeFunction': [0.750]
})



diabetes_scaler = joblib.load('scaler.pkl')

scalerInputData=diabetes_scaler.transform(test_case_5)
predictions = predicts.predict(scalerInputData)
predictions



# MLflow Tracking
MLFLOW_TRACKING_URI = "https://dagshub.com/s3akash/diseasePredictionSystemModel.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 's3akash'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '4d779774d517aa973e2858c5d93af0e87a4ca457'
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment("diseasePredictionSystemModel")

# Log to MLflow
for name, model in models.items():
    with mlflow.start_run():
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mean_cv_f1_score", cv_f1_scores.mean())
        mlflow.log_metric("std_cv_f1_score", cv_f1_scores.std())
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("true_positives", cm[1, 1])
        mlflow.log_metric("true_negatives", cm[0, 0])
        mlflow.log_metric("false_positives", cm[0, 1])
        mlflow.log_metric("false_negatives", cm[1, 0])
        
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name=name,
        )


import requests
import json

# Define test payload
test_case = {
    "Pregnancies": 8,
    "Age": 50,
    "Glucose": 180,
    "BloodPressure": 90,
    "SkinThickness": 45,
    "Insulin": 230,
    "BMI": 40.0,
    "DiabetesPedigreeFunction": 0.750
}
# Make POST request
response = requests.post("http://127.0.0.1:8000/predict", json=test_case)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())




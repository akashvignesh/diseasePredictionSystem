import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import sqlite3
from sqlite3 import Error

beforeDataClean = pd.read_csv("diabetes.csv")

afterDataClean = beforeDataClean.copy(deep = True)

#DataCleaning
afterDataClean[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = (afterDataClean[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN))
afterDataClean['Glucose'] = afterDataClean['Glucose'].fillna(round(afterDataClean['Glucose'].mean()))
afterDataClean['BloodPressure'] = afterDataClean['BloodPressure'].fillna(round(afterDataClean['BloodPressure'].mean()))
afterDataClean['SkinThickness'] = afterDataClean['SkinThickness'].fillna(round(afterDataClean['SkinThickness'].median()))
afterDataClean['Insulin'] = afterDataClean['Insulin'].fillna(round(afterDataClean['Insulin'].median()))
afterDataClean['BMI'] = afterDataClean['BMI'].fillna(round(afterDataClean['BMI'].median()))

# Check if there are any remaining NaN values
print(afterDataClean.isnull().sum())
#final data
data=afterDataClean
conn = None
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
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)
    

dbConnection=create_connection("normalized.db",True)
patientTable='''CREATE TABLE Patient (
    PatientID INTEGER  PRIMARY KEY AUTOINCREMENT ,
    Pregnancies INTEGER,
    Age INT);
'''
healthMatrixTable='''CREATE TABLE HealthMetrics (
    PatientID INTEGER,
    Glucose INTEGER,
    BloodPressure INTEGER,
    SkinThickness INTEGER,
    Insulin INTEGER,
    BMI FLOAT,
    DiabetesPedigreeFunction FLOAT,
    FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
);'''
diseaseOutcomeTable ='''CREATE TABLE Outcome (
    PatientID INTEGER,
    DiabetesStatus INTEGER,
    FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
);'''
create_table(dbConnection,patientTable)
create_table(dbConnection,healthMatrixTable)
create_table(dbConnection,diseaseOutcomeTable)

def insert_patient_data(conn, data):
    cursor = conn.cursor()
    for index, row in data.iterrows():
        cursor.execute(
            "INSERT INTO Patient (Pregnancies, Age) VALUES (?, ?)",
            (row["Pregnancies"], row["Age"])
        )
    conn.commit()

def insert_health_metrics(conn, data):
    cursor = conn.cursor()
    
    patient_ids = cursor.execute("SELECT PatientID FROM Patient").fetchall()
    
    for index, (row, patient_id) in enumerate(zip(data.iterrows(), patient_ids)):
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
    
    for index, (row, patient_id) in enumerate(zip(data.iterrows(), patient_ids)):
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



# Load the dataset
dataFromTable='''SELECT 
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
data.head()
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Outcome'], y=data['Glucose'])
plt.title('Box Plot of Glucose vs Outcome')
plt.show()

# Histograms
data.hist(bins=15, figsize=(15, 10), color='steelblue')
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()
X = data.drop('Outcome', axis=1)
y = data['Outcome']



# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
log_reg = LogisticRegression(max_iter=200, random_state=42)
svm = SVC(probability=True, random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

# Random Forest hyperparameter tuning

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Stacking
stacking = StackingClassifier(
    estimators=[('lr', log_reg), ('rf', best_rf), ('svm', svm)],
    final_estimator=LogisticRegression(max_iter=200, random_state=42)
)

# Evaluate
models = {'Logistic Regression': log_reg, 'Random Forest': best_rf, 'SVM': svm, 'Gradient Boosting': gradient_boosting, 'Stacking': stacking}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print(results_df)
   
best_model_name = results_df['Accuracy'].idxmax()
best_model = models[best_model_name]
print("Best Model is" ,best_model_name)   
# Save the best model
joblib.dump(best_rf, 'disease_prediction_model.pkl')
print("Model saved successfully.")


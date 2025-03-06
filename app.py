import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load saved models and preprocessing objects
regressor = joblib.load("best_reg_model.pkl")  # Regression Model (Duration of Stay)
classifier = joblib.load("best_clf_model.pkl")  # Classification Model (Type of Admission)
scaler = joblib.load("scaler.pkl")  # Standard Scaler
encoder_columns = joblib.load("encoder_columns.pkl")  # One-Hot Encoder column names

# Streamlit UI
st.title("Hospital Admission Prediction")
st.write("Enter patient details to predict **Duration of Stay** and **Type of Admission**.")

# Input Fields
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["M", "F"])
rural = st.selectbox("Rural Area", ["R","U"])
outcome = st.selectbox("Outcome", ["DISCHARGE", "EXPIRY", "DAMA"])
smoking = st.selectbox("Smoking", [1, 0])
alcohol = st.selectbox("Alcohol", [1, 0])
dm = st.selectbox("Diabetes Mellitus (DM)", [1, 0])
htn = st.selectbox("Hypertension (HTN)", [1, 0])
cad = st.selectbox("Coronary Artery Disease (CAD)", [1, 0])
prior_cmp = st.selectbox("Prior CMP", [1, 0])
ckd = st.selectbox("CKD", [1, 0])
raised_cardiac_enzymes = st.selectbox("Raised Cardiac Enzymes", [1, 0])
severe_anaemia = st.selectbox("Severe Anaemia", [1, 0])
anaemia = st.selectbox("Anaemia", [1, 0])
stable_angina = st.selectbox("Stable Angina", [1, 0])
stemi = st.selectbox("STEMI", [1, 0])
atypical_chest_pain = st.selectbox("Atypical Chest Pain", [1, 0])
heart_failure = st.selectbox("Heart Failure", [1, 0])
hfref = st.selectbox("HFREF", [1, 0])
hfnef = st.selectbox("HFNEF", [1, 0])
acs = st.selectbox("ACS", [1, 0])
valvular = st.selectbox("Valvular", [1, 0])
chb = st.selectbox("CHB", [1, 0])
sss = st.selectbox("SSS", [1, 0])
aki = st.selectbox("AKI", [1, 0])
cva_bleed = st.selectbox("CVA Bleed", [1, 0])
cva_infract = st.selectbox("CVA infract", [1, 0])
af = st.selectbox("AF", [1, 0])
vt = st.selectbox("VT", [1, 0])
psvt = st.selectbox("PSVT", [1, 0])
congenital = st.selectbox("Congenital", [1, 0])
uti = st.selectbox("UTI", [1, 0])
neuro_cardiogenic_syncope = st.selectbox("Neuro Cardiogenic Syncope", [1, 0])
orthostatic = st.selectbox("Orthostatic", [1, 0])
infective_endocarditis = st.selectbox("Infective Endocarditis", [1, 0])
dvt = st.selectbox("DVT", [1, 0])
cardiogenic_shock = st.selectbox("Cardiogenic Shock", [1, 0])
shock = st.selectbox("Shock", [1, 0])
pulmonary_embolism = st.selectbox("Pulmonary Embolism", [1, 0])
chest_infection = st.selectbox("Chest Infection", [1, 0])
duration_of_intensive_unit_stay = st.number_input("Duration of Intensive Unit Stay")

# Numerical Inputs
hb = st.number_input("Hemoglobin (HB)")
tlc = st.number_input("Total Leukocyte Count (TLC)")
platelets = st.number_input("Platelet Count")
glucose = st.number_input("Blood Glucose Level")
urea = st.number_input("Urea Level")
creatinine = st.number_input("Creatinine Level")
ef = st.number_input("Ejection Fraction (EF)")

# Convert Inputs into DataFrame
input_data = pd.DataFrame([[age, gender, rural, outcome, smoking, alcohol, dm, htn, cad, prior_cmp, ckd, 
                            raised_cardiac_enzymes, severe_anaemia, anaemia, stable_angina, stemi, atypical_chest_pain, 
                            heart_failure, hfref, hfnef, acs, valvular, chb, sss, aki, cva_bleed,cva_infract, af, vt, psvt, congenital, 
                            uti, neuro_cardiogenic_syncope, orthostatic, infective_endocarditis, dvt, cardiogenic_shock, shock, 
                            pulmonary_embolism, chest_infection, duration_of_intensive_unit_stay, hb, tlc, platelets, glucose, urea, creatinine, ef]], 
                          columns=["AGE", "GENDER", "RURAL", "OUTCOME", "SMOKING", "ALCOHOL", "DM", "HTN", "CAD", "PRIOR CMP", "CKD", 
                                   "RAISED CARDIAC ENZYMES", "SEVERE ANAEMIA", "ANAEMIA", "STABLE ANGINA", "STEMI", "ATYPICAL CHEST PAIN", 
                                   "HEART FAILURE", "HFREF", "HFNEF", "ACS", "VALVULAR", "CHB", "SSS", "AKI", "CVA BLEED","CVA INFRACT", "AF", "VT", "PSVT", "CONGENITAL", 
                                   "UTI", "NEURO CARDIOGENIC SYNCOPE", "ORTHOSTATIC", "INFECTIVE ENDOCARDITIS", "DVT", "CARDIOGENIC SHOCK", "SHOCK", 
                                   "PULMONARY EMBOLISM", "CHEST INFECTION", "DURATION OF INTENSIVE UNIT STAY", "HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "EF"])

# One-Hot Encoding
input_data['RURAL_U'] = (input_data['RURAL'] == 'U').astype(int)
input_data['RURAL_R'] = (input_data['RURAL'] == 'R').astype(int)
input_data['GENDER_M'] = (input_data['GENDER'] == 'M').astype(int)
input_data['GENDER_F'] = (input_data['GENDER'] == 'F').astype(int)
input_data['OUTCOME_DISCHARGE'] = (input_data['OUTCOME'] == 'DISCHARGE').astype(int)
input_data['OUTCOME_EXPIRY'] = (input_data['OUTCOME'] == 'EXPIRY').astype(int)
input_data['OUTCOME_DAMA'] = (input_data['OUTCOME'] == 'DAMA').astype(int)
input_data = input_data.drop(columns=['GENDER', 'OUTCOME'])
input_data = input_data.reindex(columns=encoder_columns, fill_value=0)

# Scale Numerical Features
numerical_features = scaler.feature_names_in_
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make Predictions
duration_of_stay = regressor.predict(input_data)[0]
type_of_admission = classifier.predict(input_data)[0]
type_of_admission_label = "Emergency" if type_of_admission == 1 else "OPD"

# Display Predictions
st.subheader("Predictions:")
st.write(f"**Predicted Duration of Stay:** {duration_of_stay:.2f} days")
st.write(f"**Predicted Type of Admission:** {type_of_admission_label}")
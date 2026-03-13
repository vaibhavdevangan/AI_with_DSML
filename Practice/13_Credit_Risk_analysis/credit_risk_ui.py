import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st

# Load data
train_input = pd.read_csv("Credit_Risk_Train_Data.csv")
validate_input = pd.read_csv("Credit_Risk_Validate_Data.csv")
validate_input.rename(columns={"outcome": "Loan_Status"}, inplace=True)
data_all = pd.concat([train_input, validate_input], ignore_index=True)

# Fill missing values as in notebook
data_all.fillna({'Gender': 'Male'}, inplace=True)
data_all.fillna({'Married': 'Yes'}, inplace=True)
bachelor_nulldependent = data_all[(data_all['Married'] == "No") & (data_all['Dependents'].isnull())].index.tolist()
data_all.loc[bachelor_nulldependent, 'Dependents'] = '0'
data_all.loc[data_all['Dependents'].isnull(), 'Dependents'] = "1"
data_all.fillna({'Self_Employed': 'No'}, inplace=True)
data_all.loc[(data_all['LoanAmount'].isnull()) & (data_all['Loan_Amount_Term'] == 360), 'LoanAmount'] = 144
data_all.loc[(data_all['LoanAmount'].isnull()) & (data_all['Loan_Amount_Term'] == 480), 'LoanAmount'] = 137
data_all.loc[data_all['LoanAmount'].isnull(), 'LoanAmount'] = 130
data_all.loc[data_all['Loan_Amount_Term'].isnull(), 'Loan_Amount_Term'] = 360
data_all.fillna({'Credit_History': 1}, inplace=True)

# Encode Dependents
data_all.loc[data_all['Dependents'] == '3+', 'Dependents'] = '3'
encoder = LabelEncoder()
data_all['Dependents'] = encoder.fit_transform(data_all['Dependents'])

# Get dummies
data_all_new = pd.get_dummies(data_all.drop(['Loan_ID'], axis=1), drop_first=True, dtype=int)

# Split features and target
X = data_all_new.drop(['Loan_Status_Y'], axis=1)
y = data_all_new['Loan_Status_Y']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Print accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f'Accuracy of Logistic Regression is {accuracy*100:.2f}%')

# Save model and scaler
with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Function to get feature names
feature_names = X.columns.tolist()

# Streamlit UI
st.title("Credit Risk Prediction")

# Inputs
gender = st.selectbox("Gender", ['Male', 'Female'], index=0)
married = st.selectbox("Married", ['Yes', 'No'], index=0)
dependents = st.selectbox("Dependents", ['0', '1', '2', '3'], index=0)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'], index=0)
self_employed = st.selectbox("Self_Employed", ['No', 'Yes'], index=0)
applicant_income = st.number_input("ApplicantIncome", min_value=0.0)
coapplicant_income = st.number_input("CoapplicantIncome", min_value=0.0)
loan_amount = st.number_input("LoanAmount", min_value=0.0)
loan_amount_term = st.number_input("Loan_Amount_Term", min_value=0.0)
credit_history = st.selectbox("Credit_History", [0, 1], index=1)
property_area = st.selectbox("Property_Area", ['Urban', 'Semiurban', 'Rural'], index=0)

if st.button("Predict Eligibility"):
    try:
        # Create input df
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [int(dependents)],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        # Preprocess
        input_data['Dependents'] = encoder.transform(input_data['Dependents'].astype(str))
        input_dummies = pd.get_dummies(input_data, drop_first=True, dtype=int)

        # Ensure all columns match
        for col in feature_names:
            if col not in input_dummies.columns:
                input_dummies[col] = 0
        input_dummies = input_dummies[feature_names]

        # Scale
        input_scaled = scaler.transform(input_dummies)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"Eligible! Probability: {prob:.2f}")
        else:
            st.error(f"Not Eligible. Probability: {prob:.2f}")

    except Exception as e:
        st.error(str(e))
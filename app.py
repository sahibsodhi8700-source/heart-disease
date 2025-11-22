import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href= f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href
# -----------------------------
# App title and tabs
# -----------------------------
st.title("❤️ Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# -----------------------------
# Tab 1: Single prediction
# -----------------------------
with tab1:
    st.header("Enter Patient Data")

    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # -----------------------------
    # Convert categorical to numeric
    # -----------------------------
    sex = 0 if sex == "Male" else 1
    chest_pain = ["typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # -----------------------------
    # Create a DataFrame
    # -----------------------------
    input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})


# -----------------------------
# Model names
# -----------------------------
algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
modelnames = ['DecisionTreeR.pkl', 'LogisticR.pkl', 'randomforestR.pkl', 'svmR.pkl']

# -----------------------------
# Prediction function
# -----------------------------
def predict_heart_disease(data):
    predictions = []
    for modelname in modelnames:
        if not os.path.exists(modelname):
            st.warning(f"Model file {modelname} not found!")
            predictions.append(None)
            continue

        try:
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
            pred = model.predict(data)
            predictions.append(pred)
        except Exception as e:
            st.error(f"Error loading {modelname}: {e}")
            predictions.append(None)
    return predictions

# -----------------------------
# Run prediction on button click
# -----------------------------
if st.button("Submit"):
    st.subheader("Results")
    st.markdown('-------------------------')

    result = predict_heart_disease(input_data)

    for i in range(len(result)):
        st.subheader(algonames[i])
        if result[i] is None:
            st.write("Prediction unavailable due to missing model.")
        elif result[i][0] == 0:
            st.write("✅ No heart disease detected.")
        elif result[i][0] == 1:
            st.write("⚠️ Heart disease detected.")
        else:
            st.write("Prediction unavailable.")
        st.markdown('-------------------------------')

# -----------------------------
# Tab 2 and Tab 3 placeholders
# -----------------------------
with tab2:
    st.header("Bulk Upload")
    st.markdown("""
    **Instructions for CSV upload:**
    1. File format: CSV (.csv)
    2. Required columns (case-sensitive): Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    3. Numeric encoding for categorical columns:
       - Sex: 0=Male, 1=Female
       - ChestPainType: 0=typical Angina, 1=Atypical Angina, 2=Non-Anginal Pain, 3=Asymptomatic
       - FastingBS: 0=<=120 mg/dl, 1=>120 mg/dl
       - RestingECG: 0=Normal, 1=ST-T Wave Abnormality, 2=Left Ventricular Hypertrophy
       - ExerciseAngina: 0=No, 1=Yes
       - ST_Slope: 0=Upsloping, 1=Flat, 2=Downsloping
    4. Make sure there are no extra blank rows or columns.
    """)
    
  
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

required_columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol',
                    'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    # Check missing columns
    missing_cols = [col for col in required_columns if col not in input_data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Keep only required columns
        input_data = input_data[required_columns]

        # Convert categorical text columns to numeric (match training)
        input_data['Sex'] = input_data['Sex'].map({'Male':0, 'Female':1})
        input_data['FastingBS'] = input_data['FastingBS'].map({'<= 120 mg/dl':0, '> 120 mg/dl':1})
        input_data['ChestPainType'] = input_data['ChestPainType'].map({
            "typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3
        })
        input_data['RestingECG'] = input_data['RestingECG'].map({
            "Normal":0, "ST-T Wave Abnormality":1, "Left Ventricular Hypertrophy":2
        })
        input_data['ExerciseAngina'] = input_data['ExerciseAngina'].map({'No':0, 'Yes':1})
        input_data['ST_Slope'] = input_data['ST_Slope'].map({"Upsloping":0, "Flat":1, "Downsloping":2})

        # Load logistic regression model
        model = pickle.load(open('LogisticR.pkl', 'rb'))

        # Predict
        input_data['Prediction LR'] = model.predict(input_data.values)

        # Show predictions
        st.subheader("Predictions")
        st.write(input_data)

        # Download link
        st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
else:
    st.info("Upload a CSV file with the required columns to get predictions.")



with tab3:
    import plotly.express as px
    data={'Decision Trees': 80.97, 'Logistic Regression':85.23,'Support vector machine':84.22,'Random Forest':86.41}
    Models=list(data.keys())
    Accuracies = list(data.values())
    df= pd.DataFrame(list(zip(Models, Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df, y='Accuracies', x='Models')
    st.plotly_chart(fig)

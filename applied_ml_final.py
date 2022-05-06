import numpy as np 
import pandas as pd 
import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Maternal Health Risk Data Set.csv')

x = df.drop(['RiskLevel'], axis=1)
y = pd.DataFrame(df['RiskLevel'])

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

x_prepared = pd.DataFrame(num_pipeline.fit_transform(x)) # scale numerical data
encoder = OrdinalEncoder()
y_prepared = pd.DataFrame(encoder.fit_transform(y))

# remember encoder key!
# 0 = high risk
# 1 = low risk
# 2 = mid risk

# Best Prediction Model from Jupyter Notebook tests
forest = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=1, random_state=50)
forest.fit(x_prepared, y_prepared)

st.title("Maternal Health Risk Predictor")
st.write("This is a machine learning resource that attempts to predict an individual's risk of health issues during pregnancy. Risk can either be high, mid, or low. This predictor takes into account factors such as age, blood pressure, blood sugar, normal body temperature, and heartrate.")
st.write("The prediction method used here is a random forest classifier algorithm. It was trained on the Maternal Health Data Set (https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data) which contains over 1,000 samples collected from different hospitals, community clinics, and maternal health care centers.")
st.write("Please provide the information below and this machine learning model will attempt to predict your risk level.") 

# gather user data
user_input = []
user_input.append(st.number_input('Enter age in years:'))
user_input.append(st.number_input('Enter upper value of blood pressure (systolic blood pressure) in mmHg:'))
user_input.append(st.number_input('Enter lower vaule of blood pressure (diastolic blood pressure) in mmHg:'))
user_input.append(st.number_input('Enter blood sugar levels in terms of molar concentration (mmol/L):'))
user_input.append(st.number_input('Enter normal body temperature ins degree Fahrenheit:'))
user_input.append(st.number_input('Enter normal resting heartrate in beats per minute:'))
    
# convert input to standardized values using mean and std values of each column in dataset
user_input[0] = (user_input[0] - 29.871795) / 13.474386 # age
user_input[1] = (user_input[1] - 113.198225) / 18.403913 # systolic bp
user_input[2] = (user_input[2] - 76.460552) / 13.885796 # diastolic bp
user_input[3] = (user_input[3] - 8.725986) / 3.293532 # blood sugar
user_input[4] = (user_input[4] - 98.665089) / 1.371384 # body temperature
user_input[5] = (user_input[5] - 74.301775) / 8.088702 # heartrate

user_input = np.array(user_input)
user_input = user_input.reshape(1, -1)

if st.button('Predict Risk'):

    # make prediction
    prediction = forest.predict(user_input)

    if prediction == 2:
        st.subheader('Result: MID RISK')
    elif prediction == 1:
        st.subheader('Result: LOW RISK')
    else:
        st.subheader('Result: HIGH RISK')

st.caption("This resource was created for academic purposes. If you are seriously concerned about your health risks, please see a real doctor.")



import joblib
import numpy as np

# load model and scaler
model = joblib.load("src/model/rf_model.pkl")
scaler = joblib.load("src/model/scaler.pkl")

# input patient data
print("Enter patient data:")
pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# pre process input data
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age]])
scaled_features = scaler.transform(features)

# predict
prediction = model.predict(scaled_features)
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")

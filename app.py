import joblib  # ✅ switched from pickle to joblib
import pandas as pd
from flask import Flask, render_template, request

# Load pre-trained model and resources
model = joblib.load('rf_model.joblib')  # ✅

label_encoders = joblib.load('label_encoders.joblib')  # ✅
le_disease = label_encoders['Disease']  # ✅

feature_order = joblib.load('feature_order.pkl')  # ✅

# Load disease-related data
df_cleaned = pd.read_csv('med_plus_cleaned.csv')

# Load and clean doctor data
df_doctors = pd.read_csv('puneDR.csv')
df_doctors.columns = df_doctors.columns.str.strip()
df_doctors['Doctor_Specialization'] = df_doctors['Specialization'].str.strip().str.lower()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    symptoms = feature_order
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    selected_symptoms = request.form.getlist("symptoms")
    age = request.form.get("age")
    gender = request.form.get("gender")
    lifestyle = request.form.get("lifestyle")

    input_dict = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in feature_order}
    input_dict["Age"] = int(age) if age else 0
    input_dict["Gender"] = gender if gender else "Unknown"
    input_dict["Lifestyle_Factors"] = lifestyle if lifestyle else "Unknown"

    input_df = pd.DataFrame([input_dict])
    safe_map = {'Male': 0, 'Female': 1, 'Unknown': 2}
    for col in ["Gender", "Lifestyle_Factors"]:
        input_df[col] = input_df[col].map(safe_map).fillna(2).astype(int)

    prediction = model.predict(input_df)[0]
    probability = round(max(model.predict_proba(input_df)[0]) * 100, 2)

    disease_name = le_disease.inverse_transform([prediction])[0]
    matched_row = df_cleaned[df_cleaned["Disease"] == disease_name].iloc[0]
    doctor = matched_row["Doctor_Specialization"]
    remedy = matched_row["Home_Remedy"]

    result = {
        "symptoms": selected_symptoms,
        "disease": disease_name,
        "probability": probability,
        "doctor": doctor,
        "remedy": remedy
    }

    return render_template("result.html", result=result)

@app.route('/doctors/<specialization>')
def show_doctors(specialization):
    specialization = specialization.strip().lower()
    df_doctors['Specialization'] = df_doctors['Specialization'].str.strip().str.lower()

    doctors = df_doctors[df_doctors['Specialization'].str.contains(specialization, case=False, na=False)]

    return render_template('doctors.html', specialization=specialization.title(), doctors=doctors.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)

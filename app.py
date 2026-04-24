import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

from med_model import (
    build_feature_frame,
    confidence_band,
    detect_red_flags,
    explain_prediction,
    normalize_symptom,
    parse_symptoms,
)


MODEL_PATH = "med_model.joblib"


def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model artifact not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)


artifact = load_model_artifact()
model = artifact["model"]
symptom_columns = artifact["symptom_columns"]
disease_profiles = artifact["disease_profiles"]

df_doctors = pd.read_csv("puneDR.csv")
df_doctors.columns = df_doctors.columns.str.strip()
df_doctors["Specialization_Normalized"] = (
    df_doctors["Specialization"].astype(str).str.strip().str.lower()
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=symptom_columns,
        metrics=artifact.get("metrics", {}),
    )


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    selected_symptoms = [
        normalize_symptom(symptom)
        for symptom in request.form.getlist("symptoms")
        if normalize_symptom(symptom)
    ]
    age = request.form.get("age") or 0
    gender = request.form.get("gender") or "Other"
    lifestyle = request.form.get("lifestyle") or "Unknown"

    record = {
        "Symptoms": selected_symptoms,
        "Age": age,
        "Gender": gender,
        "Lifestyle_Factors": lifestyle,
    }
    input_df = build_feature_frame([record], symptom_columns)
    probabilities = model.predict_proba(input_df)[0]
    selected_set = set(parse_symptoms(selected_symptoms))
    scored_predictions = []
    for index, disease in enumerate(model.classes_):
        profile = disease_profiles.get(disease, {})
        common = set(profile.get("common_symptoms", []))
        matched = selected_set & common
        symptom_precision = 0 if not selected_set else len(matched) / len(selected_set)
        symptom_recall = 0 if not common else len(matched) / min(len(common), 5)
        clinical_fit = (0.65 * symptom_precision) + (0.35 * symptom_recall)
        score = (0.65 * float(probabilities[index])) + (0.35 * clinical_fit)
        scored_predictions.append((index, score))

    ranked_indexes = [index for index, _ in sorted(scored_predictions, key=lambda item: item[1], reverse=True)[:5]]
    score_lookup = dict(scored_predictions)

    predictions = []
    for rank, index in enumerate(ranked_indexes, start=1):
        disease = model.classes_[index]
        probability = min(float(score_lookup[index]), 0.99)
        profile = disease_profiles.get(disease, {})
        explanation = explain_prediction(disease, selected_symptoms, profile)
        doctor = profile.get("doctor", "General Physician")
        predictions.append(
            {
                "rank": rank,
                "disease": disease,
                "confidence": round(probability * 100, 1),
                "confidence_value": probability,
                "confidence_band": confidence_band(probability),
                "doctor": doctor,
                "doctor_slug": doctor.lower().replace(" ", ""),
                "remedy": profile.get("remedy", "Consult a doctor for suitable care."),
                "matched_symptoms": explanation["matched_symptoms"],
                "missing_common_symptoms": explanation["missing_common_symptoms"],
                "coverage": explanation["coverage"],
            }
        )

    result = {
        "patient": {
            "age": age,
            "gender": gender,
            "lifestyle": lifestyle,
        },
        "symptoms": parse_symptoms(selected_symptoms),
        "predictions": predictions,
        "primary": predictions[0],
        "red_flags": detect_red_flags(selected_symptoms),
        "low_confidence": predictions[0]["confidence_value"] < 0.35,
        "model_metrics": artifact.get("metrics", {}),
    }

    return render_template("result.html", result=result)


@app.route("/doctors/<specialization>")
def show_doctors(specialization):
    specialization = specialization.strip().lower()
    doctors = df_doctors[
        df_doctors["Specialization_Normalized"].str.replace(" ", "", regex=False).str.contains(
            specialization,
            case=False,
            na=False,
        )
    ]

    return render_template(
        "doctors.html",
        specialization=specialization.title(),
        doctors=doctors.to_dict(orient="records"),
    )


if __name__ == "__main__":
    app.run(debug=True)

import re
from collections import Counter, defaultdict

import pandas as pd


DEMOGRAPHIC_COLUMNS = [
    "age_scaled",
    "gender_female",
    "gender_male",
    "gender_other",
    "lifestyle_active",
    "lifestyle_sedentary",
    "symptom_count_scaled",
]

URGENT_RULES = [
    {
        "label": "Possible cardiac or respiratory emergency",
        "symptoms": {"chest pain", "shortness of breath"},
        "message": "Chest pain with breathing difficulty needs urgent medical attention.",
    },
    {
        "label": "Possible stroke warning signs",
        "symptoms": {"facial drooping", "difficulty speaking"},
        "message": "Facial drooping or speech difficulty should be treated as an emergency.",
    },
    {
        "label": "Severe infection warning",
        "symptoms": {"high fever", "stiffness"},
        "message": "High fever with neck/body stiffness can be serious and needs prompt care.",
    },
    {
        "label": "Possible urinary complication",
        "symptoms": {"blood in urine", "fever"},
        "message": "Blood in urine with fever should be assessed by a doctor quickly.",
    },
]


def normalize_symptom(value):
    text = str(value or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_symptoms(value):
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value or "").split(",")
    symptoms = []
    for item in raw_items:
        symptom = normalize_symptom(item)
        if symptom:
            symptoms.append(symptom)
    return sorted(set(symptoms))


def build_symptom_columns(df):
    all_symptoms = set()
    for symptoms in df["Symptoms"].map(parse_symptoms):
        all_symptoms.update(symptoms)
    return sorted(all_symptoms)


def build_feature_frame(records, symptom_columns):
    rows = []
    for record in records:
        selected_symptoms = set(parse_symptoms(record.get("Symptoms", [])))
        gender = str(record.get("Gender", "")).strip().lower()
        lifestyle = str(record.get("Lifestyle_Factors", "")).strip().lower()
        age = pd.to_numeric(record.get("Age", 0), errors="coerce")
        age = 0 if pd.isna(age) else max(0, min(120, float(age)))

        row = {symptom: int(symptom in selected_symptoms) for symptom in symptom_columns}
        row.update(
            {
                "age_scaled": age / 120.0,
                "gender_female": int(gender == "female"),
                "gender_male": int(gender == "male"),
                "gender_other": int(gender not in {"female", "male"}),
                "lifestyle_active": int(lifestyle == "active"),
                "lifestyle_sedentary": int(lifestyle == "sedentary"),
                "symptom_count_scaled": min(len(selected_symptoms), 12) / 12.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=list(symptom_columns) + DEMOGRAPHIC_COLUMNS).fillna(0)


def build_disease_profiles(df):
    profiles = {}
    grouped = df.groupby("Disease", dropna=False)
    for disease, group in grouped:
        symptom_counter = Counter()
        for symptoms in group["Symptoms"].map(parse_symptoms):
            symptom_counter.update(symptoms)

        profiles[disease] = {
            "common_symptoms": [symptom for symptom, _ in symptom_counter.most_common(8)],
            "doctor": group["Doctor_Specialization"].mode().iloc[0],
            "remedy": group["Home_Remedy"].mode().iloc[0],
            "sample_count": int(len(group)),
        }
    return profiles


def build_specialty_map(df):
    specialty_map = defaultdict(lambda: {"diseases": set(), "symptoms": Counter()})
    for _, row in df.iterrows():
        specialty = str(row["Doctor_Specialization"]).strip()
        specialty_map[specialty]["diseases"].add(row["Disease"])
        specialty_map[specialty]["symptoms"].update(parse_symptoms(row["Symptoms"]))

    return {
        specialty: {
            "diseases": sorted(data["diseases"]),
            "common_symptoms": [symptom for symptom, _ in data["symptoms"].most_common(10)],
        }
        for specialty, data in specialty_map.items()
    }


def explain_prediction(disease, selected_symptoms, profile):
    selected = set(parse_symptoms(selected_symptoms))
    common = profile.get("common_symptoms", [])
    matched = sorted(selected & set(common))
    missing = [symptom for symptom in common if symptom not in selected][:4]
    coverage = 0 if not common else len(matched) / min(len(common), max(len(selected), 1))
    return {
        "matched_symptoms": matched,
        "missing_common_symptoms": missing,
        "coverage": round(coverage, 2),
    }


def confidence_band(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.45:
        return "Moderate"
    if probability >= 0.25:
        return "Low"
    return "Very low"


def detect_red_flags(selected_symptoms):
    selected = set(parse_symptoms(selected_symptoms))
    alerts = []
    for rule in URGENT_RULES:
        if rule["symptoms"].issubset(selected):
            alerts.append(
                {
                    "label": rule["label"],
                    "message": rule["message"],
                }
            )
    return alerts

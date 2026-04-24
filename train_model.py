import json

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from med_model import (
    build_feature_frame,
    build_disease_profiles,
    build_specialty_map,
    build_symptom_columns,
)


DATA_PATH = "MED.csv"
MODEL_PATH = "med_model.joblib"
SYMPTOMS_PATH = "symptoms.json"


def top_k_score(model, X, y, k):
    probabilities = model.predict_proba(X)
    return top_k_accuracy_score(y, probabilities, k=k, labels=model.classes_)


def train():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Symptoms", "Disease"])
    df = df.drop_duplicates().reset_index(drop=True)

    symptom_columns = build_symptom_columns(df)
    X = build_feature_frame(df.to_dict(orient="records"), symptom_columns)
    y = df["Disease"].astype(str)

    class_counts = y.value_counts()
    reusable_mask = y.map(class_counts) >= 2
    X_reusable = X[reusable_mask]
    y_reusable = y[reusable_mask]
    X_rare = X[~reusable_mask]
    y_rare = y[~reusable_mask]

    test_size = max(0.2, y_reusable.nunique() / len(y_reusable))
    X_train_common, X_test, y_train_common, y_test = train_test_split(
        X_reusable,
        y_reusable,
        test_size=test_size,
        random_state=42,
        stratify=y_reusable,
    )
    X_train = pd.concat([X_train_common, X_rare], ignore_index=True)
    y_train = pd.concat([y_train_common, y_rare], ignore_index=True)

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(
            {
                "name": name,
                "model": model,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
                "top3_accuracy": top_k_score(model, X_test, y_test, 3),
                "top5_accuracy": top_k_score(model, X_test, y_test, 5),
            }
        )

    best = max(results, key=lambda item: (item["top5_accuracy"], item["macro_f1"]))
    artifact = {
        "model": best["model"],
        "model_name": best["name"],
        "symptom_columns": symptom_columns,
        "feature_columns": list(X.columns),
        "disease_profiles": build_disease_profiles(df),
        "specialty_map": build_specialty_map(df),
        "metrics": {
            "rows": int(len(df)),
            "diseases": int(y.nunique()),
            "model": best["name"],
            "accuracy": round(best["accuracy"], 4),
            "macro_f1": round(best["macro_f1"], 4),
            "top3_accuracy": round(best["top3_accuracy"], 4),
            "top5_accuracy": round(best["top5_accuracy"], 4),
        },
    }

    joblib.dump(artifact, MODEL_PATH, compress=3)
    with open(SYMPTOMS_PATH, "w", encoding="utf-8") as file:
        json.dump(symptom_columns, file, indent=2)

    print("Training complete")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    train()

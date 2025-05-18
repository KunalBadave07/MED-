import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the raw dataset
df = pd.read_csv("MED.csv")

# Drop rows with missing Disease values
df = df.dropna(subset=['Disease'])

# Split the Symptoms string into separate symptoms
df["Symptoms"] = df["Symptoms"].str.lower().str.strip()
df["Symptoms_List"] = df["Symptoms"].str.split(',')

# Create a flat list of all unique symptoms
all_symptoms = sorted(set(sym.strip() for sublist in df["Symptoms_List"] for sym in sublist))

# One-hot encode symptoms into separate columns
for symptom in all_symptoms:
    df[symptom] = df["Symptoms"].apply(lambda x: 1 if symptom in x else 0)

# Save this list for feature order
symptom_cols = all_symptoms

# Encode Disease
le_disease = LabelEncoder()
df["Disease"] = le_disease.fit_transform(df["Disease"])

# Define X and y
X = df[symptom_cols + ["Age", "Gender", "Lifestyle_Factors"]]
y = df["Disease"]

# Encode categorical features
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})
X["Lifestyle_Factors"] = X["Lifestyle_Factors"].map({"Sedentary": 0, "Active": 1})

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X.fillna(0), y)

# Save model and encoders
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("disease_label_encoder.pkl", "wb") as f:
    pickle.dump(le_disease, f)

with open("feature_order.pkl", "wb") as f:
    pickle.dump(symptom_cols, f)

# Save cleaned dataset with human-readable values for the app
df_cleaned = df.copy()
df_cleaned["Disease"] = le_disease.inverse_transform(df["Disease"])
df_cleaned.to_csv("med_plus_cleaned.csv", index=False)

print("✅ Symptoms split and saved. Model & encoders updated.")

import pandas as pd

# Load the datasets
df_cleaned = pd.read_csv("med_plus_cleaned.csv")
df_doctors = pd.read_csv("puneDR.csv")

# Clean column names and values
df_doctors.columns = df_doctors.columns.str.strip()
df_cleaned.columns = df_cleaned.columns.str.strip()

# Normalize the specialization strings
df_doctors["Specialization"] = df_doctors["Specialization"].astype(str).str.strip().str.lower()
df_cleaned["Doctor_Specialization"] = df_cleaned["Doctor_Specialization"].astype(str).str.strip().str.lower()

# Get unique values
doctor_specializations = set(df_doctors["Specialization"].unique())
cleaned_specializations = set(df_cleaned["Doctor_Specialization"].unique())

# Find mismatches
missing = sorted(cleaned_specializations - doctor_specializations)
matches = sorted(cleaned_specializations & doctor_specializations)

# Output results
print("Specializations in med_plus_cleaned.csv not found in puneDR.csv:")
for spec in missing:
    print("-", spec)

print("\nSample matches:")
for spec in matches[:10]:  # first 10 matching ones
    print("-", spec)

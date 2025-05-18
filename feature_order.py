import pandas as pd
import pickle

# Load your cleaned dataset
df = pd.read_csv("med_plus_cleaned.csv")

# Columns to exclude from features
exclude_columns = ['Disease', 'Doctor_Specialization', 'Home_Remedy']

# Get feature columns used for training
feature_columns = [col for col in df.columns if col not in exclude_columns]

# Save feature order to a file
with open("feature_order.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("✅ feature_order.pkl saved successfully!")

import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load cleaned dataset
df = pd.read_csv("med_plus_cleaned.csv")

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Define features and target
X = df.drop(columns=["Disease", "Home_Remedy", "Doctor_Specialization"])
y = df["Disease"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Model trained with accuracy: {accuracy:.2f}%")

# Save model using joblib instead of pickle
joblib.dump(model, "rf_model.joblib")  # 🔄 this line
joblib.dump(label_encoders, "label_encoders.joblib")  # 🔄 and this one

print("✅ Model saved as 'rf_model.joblib'")
print("✅ Label encoders saved as 'label_encoders.joblib'")
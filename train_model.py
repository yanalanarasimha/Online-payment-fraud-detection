import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# Load your dataset
df = pd.read_csv("onlinefraud.csv")  # Replace with your dataset path

# Prepare data
X = df.copy()
X["type"] = X["type"].map({"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4})
X.drop(["nameOrig", "nameDest"], axis=1, inplace=True)
y = X.pop("isFraud")

# Scale
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Handle imbalance
X_train, y_train = RandomUnderSampler(sampling_strategy="majority").fit_resample(X_scaled, y)

# Train model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "rf_fraud_model.pkl")
joblib.dump(sc, "scaler.pkl")

print("Model and scaler saved successfully!")

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# 1. Load Dataset
df = pd.read_csv("dataset.csv")

# 2. Encode categorical columns & Save Encoders
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Store each encoder

# 3. Create Binary label for Anomaly Detection
df['attack_flag'] = df['class'].apply(lambda x: 0 if x == "normal" else 1)

# 4. Prepare Features (X) and Target (y)
X = df.drop(['class', 'attack_flag'], axis=1)
y = df['attack_flag']

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train Anomaly Model (Isolation Forest)
# Returns -1 for outliers, 1 for inliers
anomaly_model = IsolationForest(contamination=0.1, random_state=42)
anomaly_model.fit(X_scaled)

# 7. Train Attack Classifier (Random Forest)
# Swapped DecisionTree for RandomForest as per your report
attack_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
attack_model.fit(X_scaled, y)

# 8. SAVE EVERYTHING
joblib.dump(anomaly_model, "anomaly_model.pkl")
joblib.dump(attack_model, "attack_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl") # Save dictionary of encoders
joblib.dump(X.columns.tolist(), "feature_order.pkl")
joblib.dump(X.mean().to_dict(), "feature_means.pkl")

print("--- Training Complete with Random Forest ---")

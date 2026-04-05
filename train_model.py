import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest

df = pd.read_csv("dataset.csv")

# Encode categorical columns
categorical_cols = ['protocol_type','service','flag']

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Binary label
df['attack_flag'] = df['class'].apply(lambda x: 0 if x == "normal" else 1)

X = df.drop(['class','attack_flag'], axis=1)
y = df['attack_flag']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Anomaly Model
anomaly_model = IsolationForest(contamination=0.1)
anomaly_model.fit(X_scaled)

# Signature Model
signature_model = DecisionTreeClassifier(max_depth=6)
signature_model.fit(X_scaled,y)

# Save models
joblib.dump(anomaly_model,"model_anomaly.pkl")
joblib.dump(signature_model,"model_signature.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Hybrid IDS model trained and saved")
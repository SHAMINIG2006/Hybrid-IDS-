from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load Models and Helpers
anomaly_model = joblib.load("anomaly_model.pkl")
attack_model = joblib.load("attack_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_means = joblib.load("feature_means.pkl")
feature_order = joblib.load("feature_order.pkl")

def signature_detection(sample):
    if sample['src_bytes'] > 8000 and sample['dst_bytes'] == 0:
        return "Possible DoS Attack"
    if sample['dst_host_diff_srv_rate'] > 0.7:
        return "Probe / Scan Attack"
    return "No Signature Match"

def hybrid_ids(input_data):
    # Create DataFrame and ensure correct column order
    sample_df = pd.DataFrame([input_data])[feature_order]
    
    # 1. Scale Input
    sample_scaled = scaler.transform(sample_df)

    # 2. Anomaly Detection (Isolation Forest)
    # 1 = Normal, -1 = Anomaly
    is_anomaly = anomaly_model.predict(sample_scaled)[0]

    if is_anomaly == 1:
        return "Normal Traffic"
    else:
        # 3. Random Forest Classification
        # Predicts 0 for normal, 1 for attack based on training logic
        rf_pred = attack_model.predict(sample_scaled)[0]
        attack_label = "Malicious" if rf_pred == 1 else "Normal"
        
        sig_result = signature_detection(input_data)
        return f"Anomaly Detected → {attack_label} (via Random Forest) | Signature: {sig_result}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        user_input = {
            "src_bytes": float(request.form.get("src_bytes", 0)),
            "dst_bytes": float(request.form.get("dst_bytes", 0)),
            "dst_host_same_srv_rate": float(request.form.get("same_rate", 0)),
            "dst_host_diff_srv_rate": float(request.form.get("diff_rate", 0)),
            "dst_host_srv_count": float(request.form.get("srv_count", 0)),
            "dst_host_rerror_rate": float(request.form.get("error_rate", 0)),
            "duration": float(request.form.get("duration", 0)),
            "logged_in": float(request.form.get("logged", 0)),
            # These categorical values are handled by encoders
            "protocol_type": request.form.get("protocol_type"),
            "service": request.form.get("service"),
            "flag": request.form.get("flag")
        }

        # Fill missing features with means and Encode Categoricals
        full_sample = feature_means.copy()
        for key, value in user_input.items():
            if key in encoders: # If it's a categorical column
                try:
                    full_sample[key] = encoders[key].transform([value])[0]
                except:
                    full_sample[key] = 0 # Default if category is unknown
            else:
                full_sample[key] = value

        result = hybrid_ids(full_sample)
        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

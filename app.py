from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ================================
# LOAD TRAINED MODELS
# ================================

anomaly_model = joblib.load("anomaly_model.pkl")
attack_model = joblib.load("attack_model.pkl")

# load feature defaults
feature_means = joblib.load("feature_means.pkl")
feature_order = joblib.load("feature_order.pkl")


# ================================
# SIGNATURE BASED DETECTION
# ================================

def signature_detection(sample):

    if sample['src_bytes'] > 8000 and sample['dst_bytes'] == 0:
        return "Possible DoS Attack"

    if sample['dst_host_diff_srv_rate'] > 0.7:
        return "Probe / Scan Attack"

    if sample['dst_host_rerror_rate'] > 0.6:
        return "Connection Error Attack"

    if sample['logged_in'] == 0 and sample['dst_bytes'] < 100:
        return "Possible Login Attack"

    return "No Signature Match"


# ================================
# HYBRID IDS LOGIC
# ================================

def hybrid_ids(input_data):

    sample_df = pd.DataFrame([input_data])

    sig_result = signature_detection(input_data)

    anomaly_pred = anomaly_model.predict(sample_df)[0]

    if anomaly_pred == 0:
        return "Normal Traffic"

    else:
        attack_type = attack_model.predict(sample_df)[0]

        return f"Attack Detected → {attack_type} | Signature: {sig_result}"


# ================================
# PREPARE FULL FEATURE VECTOR
# ================================

def prepare_full_input(user_input):

    # start with dataset mean values
    sample = feature_means.copy()

    # replace with user input values
    for key in user_input:
        sample[key] = user_input[key]

    # maintain exact feature order
    ordered_sample = {k: sample[k] for k in feature_order}

    return ordered_sample


# ================================
# ROUTES
# ================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # user input from web form
        user_input = {

            "src_bytes": float(request.form["src_bytes"]),
            "dst_bytes": float(request.form["dst_bytes"]),
            "dst_host_same_srv_rate": float(request.form["same_rate"]),
            "dst_host_diff_srv_rate": float(request.form["diff_rate"]),
            "service": float(request.form["service"]),
            "dst_host_srv_count": float(request.form["srv_count"]),
            "dst_host_rerror_rate": float(request.form["error_rate"]),
            "duration": float(request.form["duration"]),
            "flag": float(request.form["flag"]),
            "logged_in": float(request.form["logged"])

        }

        # convert to full 41-feature input
        full_sample = prepare_full_input(user_input)

        # run hybrid IDS
        result = hybrid_ids(full_sample)

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {e}"


# ================================
# RUN SERVER
# ================================

if __name__ == "__main__":
    app.run(debug=True)
# 🛡️ Hybrid Intrusion Detection System (Signature + Anomaly-Based)

## 📌 Overview

The **Hybrid Intrusion Detection System (HIDS)** is a machine learning-based cybersecurity solution that combines **signature-based detection** with **anomaly detection** to identify both known and unknown network attacks.

Traditional Intrusion Detection Systems (IDS) struggle to detect zero-day attacks or generate high false positives. This project addresses these challenges by integrating rule-based detection with machine learning models, providing improved detection accuracy, better attack classification, and real-time predictions through a Flask web application.

---

## 🎯 Problem Statement

Conventional IDS solutions have several limitations:

- Signature-based IDS detects only known attacks.
- Anomaly-based IDS identifies unknown attacks but often produces false alarms.
- Many systems fail to classify attacks into meaningful categories.
- Existing solutions are difficult to deploy in real-world environments.

This project combines the strengths of both approaches to build a lightweight and practical Hybrid IDS.

---

## 🚀 Key Features

- Hybrid Signature + Machine Learning Detection
- Detection of Known and Unknown Attacks
- Multi-layer Detection Pipeline
- Real-Time Prediction using Flask
- Attack Classification
- Low False Positive Rate
- User-Friendly Web Interface
- Lightweight and Easy to Deploy

---

## 🏗️ System Architecture

The system consists of three detection layers:

### 1. Signature-Based Detection
- Detects known attacks using predefined security rules.
- Provides instant classification with minimal computation.

### 2. Anomaly Detection
- Uses **Isolation Forest** to detect abnormal network behavior.
- Identifies previously unseen attacks.

### 3. Attack Classification
- Uses **Random Forest Classifier** to categorize malicious traffic into attack types.

The final prediction is displayed through a Flask web application.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| Flask | Web Application |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Joblib | Model Serialization |
| HTML/CSS | Frontend |
| Google Colab | Model Training |
| GitHub | Version Control |

---

## 📂 Dataset

**Dataset:** NSL-KDD (KDDTest+)

- 22,544 Network Records
- 41 Features
- Binary Classification
- Multiple Attack Categories

### Attack Categories

- Normal Traffic
- DoS Attack
- Probe Attack
- Connection Error Attack
- No Signature Match

---

## 🤖 Machine Learning Models

### Isolation Forest
Used for anomaly detection to identify unknown attacks.

### Random Forest Classifier
Used for attack classification after anomaly detection.

### Rule-Based Detection
Custom security rules identify common attack patterns instantly.

---

## 🔄 Workflow

```text
User Input
      │
      ▼
Data Preprocessing
      │
      ▼
Signature Detection
      │
      ├── Known Attack
      │        │
      │        ▼
      │   Final Prediction
      │
      ▼
Isolation Forest
      │
      ▼
Random Forest Classifier
      │
      ▼
Attack Category
      │
      ▼
Flask Web Application
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | NSL-KDD |
| Records | 22,544 |
| Random Forest Accuracy | **98.14%** |
| Training Accuracy | 98.62% |
| Testing Accuracy | 98.14% |
| Detection Approach | Hybrid |

The hybrid model successfully detects both known and unknown attacks while maintaining high prediction accuracy.

---

## 💻 Application Features

- Interactive Flask Web Interface
- Real-Time Prediction
- Attack Classification
- Signature Matching
- Anomaly Detection
- Security Alert Output

---

## 📈 Results

The proposed Hybrid IDS achieved:

- High detection accuracy
- Reduced false positives
- Fast attack identification
- Detection of zero-day attacks
- Easy deployment using Flask

---

## 📁 Project Structure

```
Hybrid-Intrusion-Detection-System/
│
├── dataset/
├── models/
├── static/
├── templates/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/Hybrid-Intrusion-Detection-System.git

cd Hybrid-Intrusion-Detection-System

pip install -r requirements.txt

python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🎯 Future Improvements

- Deep Learning (LSTM/CNN)
- Live Packet Capture
- Explainable AI (SHAP/LIME)
- Docker Deployment
- Cloud Deployment (AWS/Azure)
- Interactive Dashboard
- Email & SMS Alert System

---

## 👩‍💻 Developed By

**Shamini G**

B.Tech Information Technology  
Vellore Institute of Technology (VIT), Vellore

---

## ⭐ Project Highlights

- Hybrid Intrusion Detection System
- Signature + Anomaly Detection
- Isolation Forest + Random Forest
- Flask-Based Web Application
- NSL-KDD Dataset
- Real-Time Network Threat Detection

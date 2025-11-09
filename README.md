# ⚡ WattsEye 2.0 — AI-Powered Anomaly Detection Dashboard

**WattsEye 2.0** is a production-ready **Streamlit web app** for detecting electricity-usage anomalies such as theft, tampering, or abnormal loads.  
It combines an **Isolation Forest machine-learning model** with explainable **rule-based logic** to help utilities and energy auditors quickly identify suspicious consumption patterns.

---

### 🚀 **Features**
- Loads a pretrained `IsolationForest` model (`watts_eye_iforest.pkl`), or retrains one on new data.  
- Preprocesses uploaded or local CSVs → engineers features → predicts anomalies → overlays domain rules.  
- Interactive dashboard with charts, summaries, and downloadable flagged results.  
- Designed for quick comprehension — includes built-in infographic and metric panels.  
- Modular code: `WattsEyeModel.py` (training), `project.py` (Streamlit app), and a saved model file.

---

### 🧠 **Tech Stack**
- **Python 3.10+**, **Streamlit**, **scikit-learn**, **pandas**, **numpy**, **matplotlib**, **joblib**

---

### ⚙️ **Run Locally**
```bash
# 1️⃣  Clone the repository
git clone https://github.com/<your-username>/WattsEye2.0.git
cd WattsEye2.0

# 2️⃣  Install dependencies
pip install -r requirements.txt

# 3️⃣  Launch Streamlit app
streamlit run project.py
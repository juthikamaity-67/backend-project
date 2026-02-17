from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# Load objects directly (assuming you fixed the notebook save logic)
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if isinstance(data, list):
            data = data[0]

        feature_names = ["age", "sex", "education", "currentSmoker", "cigsPerDay", 
                         "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", 
                         "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        
        # Extract features
        features = [float(data.get(name, 0)) for name in feature_names]
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)

        # FIX for Multi-output KNN: Your notebook shows KNN returns a list
        # knn.predict_proba(...) returns [array_for_target, array_for_severity]
        probs = model.predict_proba(scaled_features)
        
        # Access the first element [0] for the "Target" (disease) probability
        # and index [1] for the positive class (disease present)
        risk_percent = round(float(probs[0][0][1]) * 100, 2)

        # Severity logic
        if risk_percent >= 80:
            severity, color = "High (Critical)", "red"
        elif risk_percent >= 40:
            severity, color = "Moderate", "orange"
        else:
            severity, color = "Low", "green"

        # Actual prediction result
        prediction = model.predict(scaled_features)
        target_text = "Heart Disease Detected" if prediction[0][0] == 1 else "No Heart Disease Detected"

        return jsonify({
            "target_text": target_text,
            "risk_percentage": f"{risk_percent}%",
            "severity": severity,
            "result_color": color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Modeli yükle
try:
    model = joblib.load("model.joblib")
except Exception as e:
    model = None
    print(f"Model yüklenerken xeta bes verdi: {e}")

EXPECTED_FEATURES = [
    "island", "bill_length_mm", "bill_depth_mm", 
    "flipper_length_mm", "body_mass_g", "sex"
]

@app.route('/health', methods=['GET'])
def health_check():
    """API sağlamliq kontrolu uç noqtesi"""
    if model is None:
        return jsonify({"status": "unhealthy", "message": "Model could not be loaded."}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Tehmin uç noqtesi"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Girdi doğrulaması
    missing_features = [f for f in EXPECTED_FEATURES if f not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400
    
    # Validations for types and expected values
    valid_islands = ["Torgersen", "Biscoe", "Dream"]
    valid_sex = ["Male", "Female"]
    numeric_fields = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    
    if data.get("island") not in valid_islands:
        return jsonify({"error": f"Invalid 'island'. Expected one of {valid_islands}"}), 400
    
    if data.get("sex") not in valid_sex:
        return jsonify({"error": f"Invalid 'sex'. Expected one of {valid_sex}"}), 400
    
    for field in numeric_fields:
        val = data.get(field)
        if not isinstance(val, (int, float)):
            return jsonify({"error": f"Invalid type for '{field}'. Must be a number."}), 400
    
    try:
       
        df_input = pd.DataFrame([data])
        

        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        classes = model.classes_
        
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        return jsonify({
            "predicted_species": prediction,
            "probabilities": prob_dict
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Modeli yükle
try:
    model = joblib.load("model.joblib")
except Exception as e:
    model = None
    print(f"Model yüklenirken hata oluştu: {e}")

EXPECTED_FEATURES = [
    "island", "bill_length_mm", "bill_depth_mm", 
    "flipper_length_mm", "body_mass_g", "sex"
]

@app.route('/health', methods=['GET'])
def health_check():
    """API sağlık kontrolü uç noktası"""
    if model is None:
        return jsonify({"status": "unhealthy", "message": "Model could not be loaded."}), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin uç noktası"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Girdi doğrulaması
    missing_features = [f for f in EXPECTED_FEATURES if f not in data]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400
    
    try:
        # DataFrame'e dönüştür (Pipeline sütun isimlerini bekler)
        df_input = pd.DataFrame([data])
        
        # Tahmin ve olasılıklar
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

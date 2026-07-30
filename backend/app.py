from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# The model is expected to be in the same directory as app.py inside the container
model_path = "superkart_model.joblib"
model = None

def load_model():
    global model
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("SuperKart model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file not found at {os.path.abspath(model_path)}")

# Initial load attempt
load_model()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "message": "SuperKart Sales Prediction Backend API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        load_model() # Try reloading if it was missing initially
    
    if model is None:
        return jsonify({"success": False, "error": "Model file is missing on server."}), 500

    try:
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data] if isinstance(json_data, dict) else json_data)
        predictions = model.predict(input_df)
        return jsonify({"success": True, "predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    global model
    if model is None:
        load_model()
    
    if model is None:
        return jsonify({"success": False, "error": "Model file is missing on server."}), 500

    try:
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame(json_data)
        predictions = model.predict(input_df)
        return jsonify({"success": True, "predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

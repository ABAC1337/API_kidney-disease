from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import joblib
import logging
from datetime import datetime

# Load model
model = joblib.load('svm_fuzzy_model.pkl')

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Init Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route('/')
def home():
    return jsonify({"message": "Test API"})


@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict Kidney Disease Classification',
    'description': 'Menerima input fitur kategori dan mengembalikan hasil klasifikasi penyakit ginjal.',
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'haemoglobin_cat': {'type': 'integer', 'example': 2},
                    'specific_gravity_cat': {'type': 'integer', 'example': 1},
                    'albumin_cat': {'type': 'integer', 'example': 3},
                    'blood_glucose_random_cat': {'type': 'integer', 'example': 1},
                    'sugar_cat': {'type': 'integer', 'example': 0},
                    'age_cat': {'type': 'integer', 'example': 2},
                    'blood_urea_cat': {'type': 'integer', 'example': 2},
                    'blood_pressure_cat': {'type': 'integer', 'example': 1},
                    'serum_creatinine_cat': {'type': 'integer', 'example': 2},
                    'sodium_cat': {'type': 'integer', 'example': 1}
                },
                'required': [
                    'haemoglobin_cat', 'specific_gravity_cat', 'albumin_cat',
                    'blood_glucose_random_cat', 'sugar_cat', 'age_cat',
                    'blood_urea_cat', 'blood_pressure_cat', 
                    'serum_creatinine_cat', 'sodium_cat'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': True},
                    'predicted_label': {'type': 'string', 'example': 'Positif CKD'},
                    'probabilities': {
                        'type': 'object',
                        'properties': {
                            'Negatif CKD': {'type': 'number', 'example': 0.1345},
                            'Positif CKD': {'type': 'number', 'example': 0.8655}
                        }
                    },
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:34:56.789123'}
                }
            }
        },
        500: {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': False},
                    'error': {'type': 'string', 'example': 'Internal server error'}
                }
            }
        }
    }
})
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")

        # Ambil nilai fitur dari body
        values = np.array([[ 
            int(data['haemoglobin_cat']),
            int(data['specific_gravity_cat']),
            int(data['albumin_cat']),
            int(data['blood_glucose_random_cat']),
            int(data['sugar_cat']),
            int(data['age_cat']),
            int(data['blood_urea_cat']),
            int(data['blood_pressure_cat']),
            int(data['serum_creatinine_cat']),
            int(data['sodium_cat'])
        ]])

        # Prediksi
        reshaped = values.reshape(1, -1)
        predicted = model.predict(reshaped)[0]
        probabilities = model.predict_proba(reshaped)[0]

        label_classes = ['Not CKD', 'CKD']

        return jsonify({
            "success": True,
            'predicted_label': label_classes[predicted],
            'probabilities': {
                label_classes[0]: round(float(probabilities[0]), 4),
                label_classes[1]: round(float(probabilities[1]), 4)
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

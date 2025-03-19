from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load your demo ML model
model = tf.keras.models.load_model('apun_ka_flood_prediction_model_1.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Preprocess the image
        image = Image.open(file.stream).convert('RGB').resize((224, 224))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)
        result = "Flood Detected 🌊" if prediction[0] > 0.5 else "No Flood Detected ✅"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
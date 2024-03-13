from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from colorthief import ColorThief
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img = file.stream
        color_thief = ColorThief(img)
        dominant_color = color_thief.get_color(quality=1)

        # Predict using the model
        predictions = model.predict(np.array([dominant_color]))

        # Map predictions to categories
        categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
        result = categories[np.argmax(predictions)]

        return jsonify({'result': result})

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': "No file part in the request"})
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': "No selected file"})
        
        img = file.stream
        color_thief = ColorThief(img)
        dominant_color = color_thief.get_color(quality=1)

        # Predict using the model
        predictions = model.predict(np.array([dominant_color]))

        # Map predictions to categories
        categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
        predicted_color = categories[np.argmax(predictions)]

        return jsonify({'predicted_color': predicted_color})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

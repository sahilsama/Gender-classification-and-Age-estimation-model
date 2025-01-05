from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the custom Mean Absolute Error (mae) function
def mae(y_true, y_pred):
    """
    Custom Mean Absolute Error function.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Load the model with the custom metric
MODEL_PATH = './model/your_model.h5'
try:
    custom_objects = {'mae': mae}  # Use the custom mae function
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully!")
    print(f"Model expects input shape: {model.input_shape}")  # Debugging output
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Preprocessing function
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocesses an image for models expecting grayscale input.
    Resizes the image, converts it to grayscale, and normalizes pixel values.
    """
    try:
        # Load the image and convert to grayscale
        image = load_img(image_path, target_size=target_size, color_mode="grayscale")
        image = img_to_array(image)  # Convert to numpy array
        image = np.expand_dims(image, axis=-1)  # Add grayscale channel
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values to [0, 1]
        print(f"Preprocessed image shape: {image.shape}")  # Debugging output
        return image
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise e


# Route: Home (renders the HTML page)
@app.route('/')
def index():
    """
    Serves the HTML upload form.
    """
    return render_template('index.html')

# Route: Prediction (handles image upload and prediction)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles file upload, processes the image, and returns predictions.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(filepath)
        print(f"File saved at {filepath}")

        # Preprocess the image
        image = preprocess_image(filepath)
        print(f"Preprocessed image shape: {image.shape}")

        # Make predictions
        predictions = model.predict(image)
        print(f"Raw predictions: {predictions}")

        # Parse predictions
        gender_prob = predictions[0][0]
        age = int(predictions[1][0])

        # Interpret gender
        gender = 'Male' if gender_prob < 0.5 else 'Female'

        print(f"Predicted age: {age}, gender: {gender}")
        return jsonify({'age': age, 'gender': gender})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    app.run(debug=True)

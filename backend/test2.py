from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError

# Define the path to your model
MODEL_PATH = './model/your_model.h5'

# Add custom_objects to map 'mae' to the actual MeanAbsoluteError class
try:
    custom_objects = {'mae': MeanAbsoluteError()}  # Map 'mae' to the correct Keras metric
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print(f"Model loaded successfully! Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")

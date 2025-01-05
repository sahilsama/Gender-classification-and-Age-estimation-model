from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError

# Map 'mae' to its corresponding Keras metric
custom_objects = {'mae': MeanAbsoluteError()}

try:
    model = load_model('./model/your_model.h5', custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

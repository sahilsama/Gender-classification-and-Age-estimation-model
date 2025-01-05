# Step 1: Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Define the Mean Absolute Error (mae) function
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Step 2: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Step 3: Load the saved model with the custom metric
model = load_model('./model/your_model.h5', custom_objects={'mae': mae})
print("Model loaded successfully!")

# Step 4: Prepare the input data
image_path = '/content/drive/My Drive/UTFface/crop_part1/78_0_0_20170111222500159.jpg.chip.jpg'  # Replace with your image path
img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')  # Match model input
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if needed

# Step 5: Make predictions
predictions = model.predict(img_array)
print("Predictions:", predictions)

# Step 6: Interpret the output
predicted_class = np.argmax(predictions, axis=-1)
print("Predicted Class:", predicted_class)

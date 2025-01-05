# Age and Gender Prediction Web App
This is a Flask-based web application that predicts the age and gender of a person from an uploaded image using a deep learning model. It was developed as a 5th Semester Mini Project to demonstrate the integration of machine learning and web development.

# 📋 Table of Contents
## About the Project🌟
This project leverages a deep learning model trained to predict age and gender. It features:

A modern and responsive web interface created with Bootstrap.
A pre-trained model integrated into the Flask app to process uploaded images and provide predictions.
Efficient preprocessing, including grayscale conversion, resizing, and normalization.


## ✨ Features
Upload an image to get predictions for:
Age
Gender (Male/Female)
Clean and sleek web design with a dark theme and interactive buttons.
Responsive and user-friendly interface.
Lightweight backend using Flask and TensorFlow.
## 🛠 Technologies Used
### Frontend
HTML5
CSS3
Bootstrap 5
### Backend
Flask: Lightweight Python web framework.
TensorFlow: Used to load and run the deep learning model.
Keras: For model building and training.
Other Tools
Pillow (PIL): For image preprocessing.
Numpy: For array manipulation.
Python: Main programming language.
## 🚀 Setup Instructions
### Follow these steps to set up and run the project locally:

1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/age-gender-prediction
cd age-gender-prediction
2. Install Dependencies
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
3. Add Your Model
Place the pre-trained TensorFlow model (your_model.h5) in the model/ directory.

4. Run the Application
Start the Flask server:

bash
Copy code
python app.py
5. Open in Browser
Visit http://127.0.0.1:5000/ in your browser to use the app.

## 🖥 Usage
Open the web app in your browser.
Upload an image using the file input field.
Click on the Predict button.
The app will display the predicted age and gender.
## 🔮 Future Enhancements
Add more features like emotion recognition or ethnicity prediction.
Improve the model's accuracy with a larger and more diverse dataset.
Deploy the app on platforms like Heroku or AWS for global accessibility.
Add support for batch predictions (multiple images at once).

## 📝 License
Under MIT License For educational purposes only.

import numpy as np
from flask import Flask, request, render_template
from waitress import serve

# Create a Flask web application
flask_app = Flask(__name__)

# Load the trained machine learning model from the pickle file
# 'rb' stands for 'read binary'
import pickle
model = pickle.load(open("ML_Model.pkl", "rb"))

# Route for the home page ('/') — what users see when they visit your website
@flask_app.route("/")
def Home():
    # It renders an HTML page called 'index.html' (stored in the 'templates' folder)
    return render_template("index.html")

# Route for the prediction logic, triggered when user submits the form
@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Extracts input values from the form (which are strings), converts them to float
    float_features = [float(x) for x in request.form.values()]

    # Converts the list to a NumPy array and wraps it in another list to match model input shape
    features = [np.array(float_features)]

    # Predict the crop using the trained model
    prediction = model.predict(features)

    # Renders the same HTML page but now with the prediction result added to it
    return render_template("index.html", Prediction_Text = f"The Predicted Crop is {prediction[0].capitalize()}.")

# Starts the app using Waitress (a WSGI server) when this file is run directly
if __name__ == "__main__":
    serve(flask_app, host="0.0.0.0", port=8000)

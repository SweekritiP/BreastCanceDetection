import numpy as np
from flask import Flask, request, render_template
import pickle
from custom_kmeans import CustomKMeans

app = Flask(__name__)

# Load the scaler and the model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict using the model directly
def predict_with_model(input_data):
    # Ensure the input data is a NumPy array
    input_data = np.array(input_data)

    # Scale the input data using the same scaler
    scaled_data = scaler.transform(input_data)

    # Predict using the model
    return model.predict(scaled_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming you have a form with input fields for each feature
    input_features = [float(x) for x in request.form.values()]
    features_array = np.array(input_features).reshape(1, -1)  # Reshape to 2D array if needed

    # Make prediction using the model
    prediction = predict_with_model(features_array)[0]

    # Display the cluster number directly
    result = f"Cluster number: {prediction}"

    return render_template('index.html', prediction_text=f'Patient belongs to {result}')

if __name__ == "__main__":
    app.run(debug=True)

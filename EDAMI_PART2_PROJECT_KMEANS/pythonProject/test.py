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

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming you have a form with input fields for each feature
    input_features = [float(x) for x in request.form.values()]
    features_array = np.array(input_features).reshape(1, -1)  # Reshape to 2D array if needed

    # Scale the input features
    features_array_scaled = scaler.transform(features_array)

    # Make prediction using the loaded model
    prediction = model.predict(features_array_scaled)[0]

    # Map prediction to a human-readable label
    if prediction == 1:
        result = "Breast cancer"
    else:
        result = "No breast cancer"

    return render_template('index.html', prediction_text=f'Patient has {result}')

if __name__ == "__main__":
    app.run(debug=True)
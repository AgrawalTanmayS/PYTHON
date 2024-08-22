from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create a Flask app
app = Flask(__name__)

# Function to get prediction
def get_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        pedigree = float(request.form['pedigree'])
        age = float(request.form['age'])

        # Get prediction
        result = get_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age)

        return render_template('result.html', result='has diabetes' if result == 1 else 'does not have diabetes')
    except Exception as e:
        return render_template('error.html', error_message=str(e))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

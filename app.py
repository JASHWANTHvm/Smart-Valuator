from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_features = [float(request.form.get(feature)) for feature in [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
        ]]

        # Scale input
        input_scaled = scaler.transform([input_features])

        # Predict house price
        predicted_price = model.predict(input_scaled)[0]

        return render_template('result.html', price=round(predicted_price, 2))
    
    except Exception as e:
        return render_template('result.html', price="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)

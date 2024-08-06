from flask import Flask, request, jsonify, render_template
import joblib  # Or use pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data[key]) for key in data]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

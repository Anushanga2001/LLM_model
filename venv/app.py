from flask import Flask, request, jsonify, render_template
import joblib  # Or use pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data[key]) for key in data]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

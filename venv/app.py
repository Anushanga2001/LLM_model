from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load the pickled model
model_path = 'model/model.pkl'

def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model: {str(e)}')

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400

        if hasattr(model, 'predict_next_words'):
            prediction = model.predict_next_words(sentence)
        else:
            return jsonify({'error': 'Model does not have the required method'}), 500

        return jsonify({'prediction': prediction})
    except Exception as e:
        app.logger.error(f'Error during prediction: {e}') 
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
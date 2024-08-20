from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import traceback  # Import traceback for detailed error reporting

app = Flask(__name__)
# get the flask 

# Load the trained model
try:
    model = load_model('model/next_word_prediction.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()  # Print the stack trace for the model loading error

# Route to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if 'input' is provided and not None
        if 'input' not in data or data['input'] is None:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert input data to numpy array
        input_data = np.array(data['input'])

        # Check if input_data is valid and not empty
        if input_data.size == 0:
            return jsonify({'error': 'Input data is empty or invalid'}), 400

        # Reshape the input data to match the model's expected input shape
        input_data = input_data.reshape((1, input_data.shape[0], 1))  # Modify as necessary

        # Print the reshaped input data for debugging purposes
        print("Input data shape:", input_data.shape)

        # Make a prediction using the model
        prediction = model.predict(input_data)
        print("Prediction made successfully:", prediction)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()  # Print the stack trace for the error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
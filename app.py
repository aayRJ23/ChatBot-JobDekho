from flask import Flask, request, jsonify, render_template
import pandas as pd
from flask_cors import CORS
import pickle
import os

# Create Flask app — template_folder='.' so index.html is found in the same directory
app = Flask(__name__, template_folder='.')

# Enable CORS for all routes (allows React frontend on port 5173 to call this)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model — use absolute path so it works regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'job_portal_model.pkl'), 'rb'))


@app.route('/', methods=['GET'])
def get_data():
    """Standalone HTML test page (for development/testing only)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts: POST JSON { "User Input": "some question" }
    Returns: JSON { "Prediction": "bot response" }
             or   { "error": "message" }
    """
    try:
        data = request.get_json()

        # Validate input
        if not data or 'User Input' not in data:
            return jsonify({'error': 'Missing "User Input" key in request body'}), 400

        user_input = data['User Input'].strip()

        if not user_input:
            return jsonify({'error': 'User Input cannot be empty'}), 400

        # Predict using the trained pipeline
        query_df = pd.DataFrame([user_input], columns=['User Input'])
        prediction = model.predict(query_df['User Input'])

        return jsonify({'Prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved models
with open('shannon_model.pkl', 'rb') as f:
    shannon_model = pickle.load(f)
with open('carbon_model.pkl', 'rb') as f:
    carbon_model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Forest Diversity & Carbon API! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        shannon_pred = float(shannon_model.predict(features)[0])
        carbon_pred = float(carbon_model.predict(features)[0])
        return jsonify({
            'shannon_prediction': shannon_pred,
            'carbon_prediction': carbon_pred,
            'note': 'These predictions are based on your submitted forest inventory data.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

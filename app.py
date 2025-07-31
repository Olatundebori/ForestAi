{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3f7db08-cc8e-4e99-8263-8a0001b0460d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the saved models\n",
    "with open('shannon_model.pkl', 'rb') as f:\n",
    "    shannon_model = pickle.load(f)\n",
    "with open('carbon_model.pkl', 'rb') as f:\n",
    "    carbon_model = pickle.load(f)\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return \"Welcome to the Forest Diversity & Carbon API! Use the /predict endpoint.\"\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        data = request.get_json(force=True)\n",
    "        # Example: data['features'] = [dbh, height, species_count, ...]\n",
    "        features = np.array(data['features']).reshape(1, -1)\n",
    "\n",
    "        shannon_pred = float(shannon_model.predict(features)[0])\n",
    "        carbon_pred = float(carbon_model.predict(features)[0])\n",
    "\n",
    "        return jsonify({\n",
    "            'shannon_prediction': shannon_pred,\n",
    "            'carbon_prediction': carbon_pred,\n",
    "            'note': 'These predictions are based on your submitted forest inventory data.'\n",
    "        })\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)}), 400\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import traceback  

app = Flask(__name__)
CORS(app)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

nn_model = tf.keras.models.load_model('nn_model.h5')
dnn_model = tf.keras.models.load_model('dnn_model.h5')

label_mapping = {
    0: 'open',
    1: 'not a real question',
    2: 'off topic',
    3: 'not constructive',
    4: 'too localized'
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Extract the text and model type from the incoming JSON
        text = data['text']
        model_type = data.get('model', 'nn')  # Default to 'nn' if not provided

        # Input validation: Check if text is provided
        if not text:
            return jsonify({'error': 'Text input is empty'}), 400

        # Preprocess the text using the loaded vectorizer
        text_tfidf = vectorizer.transform([text])

        # Check if the vectorized text is empty
        if text_tfidf.shape[0] == 0:
            return jsonify({'error': 'Text is invalid or not properly vectorized'}), 400

        # Choose the model based on the input model type (nn or dnn)
        if model_type == 'nn':
            model = nn_model
        elif model_type == 'dnn':
            model = dnn_model
        else:
            return jsonify({'error': 'Invalid model type. Choose either "nn" or "dnn".'}), 400

        text_tfidf = text_tfidf.toarray()  
        prediction = model.predict(text_tfidf)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = int(predicted_class_idx)  

        predicted_label = label_mapping.get(predicted_class, 'Unknown')

        return jsonify({
            'prediction_class_number': predicted_class,
            'prediction_label': predicted_label
        })

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error occurred: {error_message}")
        print(f"Stack trace: {stack_trace}")
        return jsonify({'error': f'An error occurred: {error_message}', 'stack_trace': stack_trace}), 500

if __name__ == '__main__':
    app.run(debug=True)

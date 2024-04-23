from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import pickle
import joblib
from keras.models import model_from_json

app = Flask(__name__)

max_length = 140


# Load Twitter model
with open('model_cudnn_lstm_architecture2.joblib', 'rb') as f:
    model_twitter = model_from_json(joblib.load(f))

model_twitter.load_weights('model_cudnn_lstm_weights2.h5')

with open('Tokenizer2.pickle', 'rb') as handle:
    tokenizer_twitter = pickle.load(handle)


# Load Combined model
with open('model_cudnn_lstm_architectureC.joblib', 'rb') as f:
    model_combined = model_from_json(joblib.load(f))

model_combined.load_weights('model_cudnn_lstm_weightsC.h5')

with open('Tokenizer_combined.pickle', 'rb') as handle:
    tokenizer_combined = pickle.load(handle)


def predict_with_model(user_input, model, tokenizer, max_length, threshold=0.5):
    
    # Tokenize and preprocess user input
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length)

    # Make prediction
    prediction_prob = model.predict(user_input_padded)

    # Work the certaintly of the model
    probability_sarcastic = float(prediction_prob[0, 0])
    probability_not_sarcastic = 1.0 - probability_sarcastic

    if prediction_prob[0, 0] > threshold:
        #predicted_label = 1
        predicted_label = "The model predicts that the input is sarcastic."
        certainty = f"{probability_sarcastic * 100:.2f}%"

    else:
        #predicted_label = 0
        predicted_label = "The model predicts that the input is not sarcastic."
        certainty = f"{probability_not_sarcastic * 100:.2f}%"
    return predicted_label, certainty



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_input/<model_type>')
def model_input(model_type):
    if model_type not in ['twitter', 'combined']:
        return "Invalid model type", 404
    return render_template('ModelInput.html', model_type=model_type)


@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model_type']
    user_input = request.form['user_input']
    
    if model_type == 'twitter':
        loaded_model, tokenizer = model_twitter, tokenizer_twitter
    elif model_type == 'combined':
        loaded_model, tokenizer = model_combined, tokenizer_combined
    else:
        return "Invalid model type", 400
    
    predicted_label, certainty = predict_with_model(user_input, loaded_model, tokenizer, max_length)
    return render_template('result.html', user_input=user_input, prediction=predicted_label, certainty=certainty)



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

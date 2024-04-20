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

# Load the Keras model
with open('model_cudnn_lstm_architecture2.joblib', 'rb') as f:
    m1_loaded = model_from_json(joblib.load(f))

m1_loaded.load_weights('model_cudnn_lstm_weights2.h5')

with open('Tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def predict_sarcasm(user_input, model, tokenizer, max_length, threshold=0.5):
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

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    predicted_label, certainty = predict_sarcasm(user_input,m1_loaded,tokenizer, max_length, threshold=0.5)
    

    return render_template('result.html', prediction=predicted_label, certainty=certainty)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

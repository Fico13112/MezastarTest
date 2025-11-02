from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import librosa
import os

app = Flask(__name__)

# Load your Keras model
model = tf.keras.models.load_model("pokemon_sound_model_v5.h5")

labels = ['Zeraora', 'Solgaleo', 'Lugia', 'Ho-Oh', 'Lunala', 'Greninja',
          'Eternatus', 'Keldeo', 'Grimmsnarl', 'Zygarde']

def extract_mfcc(y, sr, max_len=300):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pad = max_len - mfcc.shape[1]
    if pad > 0:
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")

@app.route("/script.js")
def js():
    return send_from_directory(".", "script.js")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    y, sr = librosa.load(f, sr=16000)
    X = extract_mfcc(y, sr)
    X = (X - np.mean(X)) / (np.std(X)+1e-6)
    X = np.expand_dims(X, axis=(0,-1))  # shape (1, 40, 300, 1)
    pred = model.predict(X)
    idx = int(np.argmax(pred))
    return jsonify({"prediction": labels[idx], "confidence": float(np.max(pred))})

if __name__ == "__main__":
    app.run(debug=True)


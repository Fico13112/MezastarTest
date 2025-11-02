from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Load model from local file
model = load_model('pokemon_sound_model_v5.keras')  # Make sure this .keras file is in the same folder
labels = ['Zeraora', 'Solgaleo', 'Lugia', 'Ho-Oh', 'Lunala', 'Greninja', 
          'Eternatus', 'Keldeo', 'Grimmsnarl', 'Zygarde']

# MFCC extraction (must match training)
def extract_mfcc(y, sr, max_len=300):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pad = max_len - mfcc.shape[1]
    if pad > 0:
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    y, sr = librosa.load(file, sr=16000)
    mfcc = extract_mfcc(y, sr)
    X = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    X = X[np.newaxis, ..., np.newaxis]  # shape (1, 40, 300, 1)

    pred = model.predict(X)
    pred_label = labels[np.argmax(pred)]

    return jsonify({'prediction': pred_label})

if __name__ == "__main__":
    app.run(debug=True)

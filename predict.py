import numpy as np
import librosa
import joblib

audio_file = input("Enter audio file path: ")

model = joblib.load("models/voice_model.pkl")

y, sr = librosa.load(audio_file, sr=None)

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=40
)

features = np.mean(mfcc.T, axis=0).reshape(1, -1)

prediction = model.predict(features)[0]

if prediction == 0:
    print("Prediction: REAL VOICE")
else:
    print("Prediction: FAKE VOICE")

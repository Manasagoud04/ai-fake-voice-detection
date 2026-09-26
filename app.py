import librosa
import numpy as np
import joblib

print("AI Fake Voice Detection Project Started!")

file_path = input("Enter audio file path: ")

audio, sample_rate = librosa.load(file_path, sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio duration:", len(audio) / sample_rate, "seconds")

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

features = np.mean(mfcc.T, axis=0).reshape(1, -1)

model = joblib.load("models/voice_model.pkl")

prediction = model.predict(features)[0]

print("MFCC features extracted successfully!")
print("MFCC shape:", mfcc.shape)

if prediction == 0:
    print("Prediction: REAL VOICE")
else:
    print("Prediction: FAKE VOICE")

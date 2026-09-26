import os
import numpy as np
import librosa
import joblib
from sklearn.metrics import accuracy_score, classification_report

model = joblib.load("models/voice_model.pkl")

X = []
y_true = []

folders = {
    "dataset/real": 0,
    "dataset/fake": 1
}

for folder, label in folders.items():
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder, filename)

            y_audio, sr = librosa.load(filepath, sr=None)

            mfcc = librosa.feature.mfcc(
                y=y_audio,
                sr=sr,
                n_mfcc=40
            )

            features = np.mean(mfcc.T, axis=0)

            X.append(features)
            y_true.append(label)

X = np.array(X)
y_true = np.array(y_true)

predictions = model.predict(X)

accuracy = accuracy_score(y_true, predictions)

print("Model Accuracy:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(
    y_true,
    predictions,
    target_names=["REAL", "FAKE"]
))

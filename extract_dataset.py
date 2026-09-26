import os
import numpy as np
import librosa

X = []
y = []

folders = {
    "data/real": 0,
    "data/fake": 1
}

for folder, label in folders.items():
    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            path = os.path.join(folder, filename)

            audio, sr = librosa.load(path, sr=16000)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=40
            )

            features = np.mean(mfcc.T, axis=0)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save("features/X_dataset.npy", X)
np.save("features/y_dataset.npy", y)

print("Dataset features created successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)

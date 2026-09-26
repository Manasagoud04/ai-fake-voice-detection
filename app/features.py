import librosa
import numpy as np

audio_path = "audio/test.wav"

y, sr = librosa.load(audio_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

features = mfcc.mean(axis=1)

print("Audio loaded successfully")
print("Sample rate:", sr)
print("Duration:", len(y) / sr, "seconds")
print("MFCC shape:", mfcc.shape)
print("Feature shape:", features.shape)
print("Features:", features)

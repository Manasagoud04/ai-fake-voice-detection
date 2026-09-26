import librosa
import numpy as np

def extract_mfcc(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


real_file = "dataset/real/test.wav"
fake_file = "dataset/fake/fake.wav"

real_features = extract_mfcc(real_file)
fake_features = extract_mfcc(fake_file)

print("Real audio features:")
print(real_features)

print("\nFake audio features:")
print(fake_features)

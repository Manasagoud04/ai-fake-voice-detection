import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

X = np.load("features/X_dataset.npy")
y = np.load("features/y_dataset.npy")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "models/voice_model.pkl")

print("Model trained successfully!")
print("Model saved to: models/voice_model.pkl")

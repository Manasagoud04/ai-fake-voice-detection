from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("models/voice_model.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["audio"]

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        audio, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "REAL VOICE"
        else:
            result = "FAKE VOICE"

        os.remove(file_path)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)

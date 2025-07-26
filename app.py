from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import pickle
import numpy as np

def predict(input_data):

    with open("models/fraud_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/columns.pkl", "rb") as f:
        columns = pickle.load(f)

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    return prediction
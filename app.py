import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
numeric_features = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'price']
categorical_features = ['Segment']

# create a preprocessor object to preprocess the data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    )

# open the model_rfr.joblib file in read mode
with open('../UsedCarsPricePrediction/models_joblib/model_rfr.joblib', 'rb') as f_in:
    model = joblib.load(f_in)

def predict_single(customer, model):
    # turn customer dict into dataframe
    df = pd.DataFrame([customer])
    # preprocess the data
    X = preprocessor.fit_transform(df)
    # make prediction
    prediction = model.predict(X)[0]
    return {'prediction': prediction}


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # get the json file
    customer = request.get_json(force=True)
    # make prediction
    results = predict_single(customer, model)
    # send back to browser
    return jsonify(results)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

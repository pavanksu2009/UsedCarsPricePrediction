import os
import json
import joblib
import tarfile

import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

# model_path for used_cars.csv from local machine
model_path = f"../UsedCarsPricePrediction/model/model.joblib"
model = joblib.load(model_path)

print("Loading test input data")
# test_data_directory for used_cars.csv from local machine
test_data_directory = "../UsedCarsPricePrediction/test"
test_features_data = os.path.join(test_data_directory, "test_features.csv")
test_labels_data = os.path.join(test_data_directory, "test_labels.csv")

X_test = pd.read_csv(test_features_data, header=None)
y_test = pd.read_csv(test_labels_data, header=None)

y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")

report_dict = {
        "regression_metrics": {
                "mse": {
                        "value": mean_squared_error(y_test, y_pred)
                },
                "rmse": {
                        "value": mean_squared_error(y_test, y_pred, squared=False)
                },
                "r2": {
                        "value": r2_score(y_test, y_pred)
                }
        }
}

# evaluation_output_path for used_cars.csv from local machine
evaluation_output_path = os.path.join("../UsedCarsPricePrediction/evaluation", "evaluation.json")

with open(evaluation_output_path, "w") as f:
      f.write(json.dumps(report_dict))

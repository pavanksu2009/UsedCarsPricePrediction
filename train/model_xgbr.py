import os
import joblib

import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# training_data_directory for used_cars.csv from local machine
training_data_directory = "../UsedCarsPricePrediction/train"
# validation_data_directory for used_cars.csv from local machine
validation_data_directory = "../UsedCarsPricePrediction/train"

train_features_data = os.path.join(training_data_directory, "train_features.csv") # this
train_labels_data = os.path.join(training_data_directory, "train_labels.csv")

val_features_data = os.path.join(validation_data_directory, "val_features.csv")
val_labels_data = os.path.join(validation_data_directory, "val_labels.csv")


X_train = pd.read_csv(train_features_data, header=None)
y_train = pd.read_csv(train_labels_data, header=None)

model_xgbr = XGBRegressor(n_estimators=100, random_state=42)

model_xgbr.fit(X_train, y_train)

X_val = pd.read_csv(val_features_data, header=None)
y_val = pd.read_csv(val_labels_data, header=None)

y_pred_val = model_xgbr.predict(X_val)

print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
# print accuracy score on the training and validation sets
print(f"Training Accuracy: {model_xgbr.score(X_train, y_train)}")
print(f"Validation Accuracy: {model_xgbr.score(X_val, y_val)}")

# model_output_directory for used_cars.csv from local machine
model_output_directory = os.path.join("../UsedCarsPricePrediction/model", "model_xgbr.joblib")

print(f"Saving model to {model_output_directory}")
joblib.dump(model_xgbr, model_output_directory)
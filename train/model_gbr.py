
# import mlflow
import argparse

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# mlflow.start_run()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to train data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    
    target = 'price'
    numeric_features = ['Segment','Kilometers_Driven', 'Mileage', 'Engine','Power','Seats']
    categorical_features = []

    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # split the training data into train and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
    )

    model_gbr = GradientBoostingRegressor(
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate
    )

    model_pipeline = make_pipeline(model_gbr)

    model_pipeline.fit(X_train, y_train)

    # print the training accuracy score
    print("training accuracy score:", model_pipeline.score(X_train, y_train))

    # print the training evaluation metrics
    print("training R2 score:", model_pipeline.score(X_train, y_train))
    print("training MAE:", mean_absolute_error(y_train, model_pipeline.predict(X_train)))
    print("training MSE:", mean_squared_error(y_train, model_pipeline.predict(X_train)))
    print("training RMSE:", np.sqrt(mean_squared_error(y_train, model_pipeline.predict(X_train))))

    # apply the model to the validation dataset 
    y_pred = model_pipeline.predict(X_val)

    # print validation accuracy and R2 score
    print("validation accuracy:", model_pipeline.score(X_val, y_val))

    # print the evaluation metrics for the validation dataset 
    print("validation R2 score:", r2_score(y_val, y_pred))
    print("validation MAE:", mean_absolute_error(y_val, y_pred))
    print("validation MSE:", mean_squared_error(y_val, y_pred))
    print("validation RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))

    # build a dataframe with evaluation metrics for training and validation datasets 
    eval_df = pd.DataFrame(
    {
    'R2_score_train': [model_pipeline.score(X_train, y_train)],
    'R2_score_val': [model_pipeline.score(X_val, y_val)],
    'MAE_train': [mean_absolute_error(y_train, model_pipeline.predict(X_train))],
    'MAE_val': [mean_absolute_error(y_val, y_pred)],
    'MSE_train': [mean_squared_error(y_train, model_pipeline.predict(X_train))],
    'MSE_val': [mean_squared_error(y_val, y_pred)],
    'RMSE_train': [np.sqrt(mean_squared_error(y_train, model_pipeline.predict(X_train)))],
    'RMSE_val': [np.sqrt(mean_squared_error(y_val, y_pred))]
    }
    )

    # save the model to the outputs directory for capture
    model_output_path = 'outputs/model_gbr.pkl'
    joblib.dump(model_pipeline, model_output_path)
    print("saved model to", model_output_path)
    
if __name__ == '__main__':
    main()

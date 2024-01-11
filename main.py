# import necessary libraries
import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Function to preprocess the data
def preprocess(input_data_path, target, numeric_features, categorical_features, train_features_output_path, train_labels_output_path, val_features_output_path, val_labels_output_path, test_features_output_path, test_labels_output_path):
    """
    This function preprocesses the data and saves the train, validation and test sets to csv files.
    """
    # read the data
    used_cars = pd.read_csv(input_data_path)

    # split the data into train and test sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(used_cars.drop(columns=[target]), used_cars[target],
                                                    test_size=0.2,
                                                    random_state=42)

    # split the train data into train and validation sets
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain,
                                                  test_size=0.2,
                                                  random_state=42)

    # create a preprocessor object to preprocess the data
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore',
                       sparse=False), categorical_features)
    )

    transformed_Xtrain = preprocessor.fit_transform(Xtrain)
    transformed_Xval = preprocessor.transform(Xval)
    transformed_Xtest = preprocessor.transform(Xtest)

    # save the validation set to csv file
    pd.DataFrame(transformed_Xval).to_csv(val_features_output_path,
                                          header=False, index=False)
    pd.DataFrame(transformed_Xtrain).to_csv(train_features_output_path,
                                            header=False, index=False)
    pd.DataFrame(transformed_Xtest).to_csv(test_features_output_path,
                                           header=False, index=False)

    ytrain.to_csv(train_labels_output_path, header=False, index=False)
    yval.to_csv(val_labels_output_path, header=False, index=False)
    ytest.to_csv(test_labels_output_path, header=False, index=False)

# Function to build a Logistic Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def training_lr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_lr = LinearRegression()

    model_lr.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_lr.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_lr.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_lr.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_lr, model_output_directory)

# Function to evaluate the Logistic Regression model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_lr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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


    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 

def training_dtr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_dt = DecisionTreeRegressor()

    model_dt.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_dt.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets 
    print(f"Training Accuracy: {model_dt.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_dt.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_dt, model_output_directory)

# Function to evaluate the Decision Tree Regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_dtr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a linear regression model with ridge regularization 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def training_lr_reg_ridge(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_lr_reg = Ridge(alpha=0.5)

    model_lr_reg.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_lr_reg.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets 
    print(f"Training Accuracy: {model_lr_reg.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_lr_reg.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_lr_reg, model_output_directory)

# Function to evaluate the linear regression model with ridge regularization
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_lr_reg_ridge(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a linear regression model with lasso regularization
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def training_lr_reg_lasso(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_lr_reg_lasso = Lasso(alpha=0.5)

    model_lr_reg_lasso.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_lr_reg_lasso.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_lr_reg_lasso.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_lr_reg_lasso.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_lr_reg_lasso, model_output_directory)

# Function to evaluate the linear regression model with lasso regularization
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_lr_reg_lasso(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a k-nearest neighbors regression model 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def training_knn(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_knn = KNeighborsRegressor(n_neighbors=5)

    model_knn.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_knn.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_knn.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_knn.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_knn, model_output_directory)

# Function to evaluate the k-nearest neighbors regression model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_knn(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def training_rfr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    model_rfr.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_rfr.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_rfr.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_rfr.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_rfr, model_output_directory)

# Function to evaluate the Random Forest Regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_rfr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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


    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a Bagging Regressor model
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

def training_br(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_br = BaggingRegressor(n_estimators=100, random_state=42)

    model_br.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_br.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_br.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_br.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_br, model_output_directory)

# Function to evaluate the Bagging Regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_br(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a ada boost regressor model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

def training_abr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_abr = AdaBoostRegressor(n_estimators=100, random_state=42)

    model_abr.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_abr.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_abr.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_abr.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_abr, model_output_directory)

# Function to evaluate the ada boost regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_abr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build an xgboost regressor model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def training_xgbr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
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

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_xgbr, model_output_directory)

# Function to evaluate the xgboost regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_xgbr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))

# Function to build a support vector regressor model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def training_svr(train_features_data, train_labels_data, val_features_data, val_labels_data, model_output_directory):
    """
    This function trains the model and saves the model to the model_output_directory.
    """
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    model_svr.fit(X_train, y_train)

    X_val = pd.read_csv(val_features_data, header=None)
    y_val = pd.read_csv(val_labels_data, header=None)

    y_pred_val = model_svr.predict(X_val)

    print(f"RMSE: {mean_squared_error(y_val, y_pred_val, squared=False)};")
    # print accuracy score on the training and validation sets
    print(f"Training Accuracy: {model_svr.score(X_train, y_train)}")
    print(f"Validation Accuracy: {model_svr.score(X_val, y_val)}")

    print(f"Saving model to {model_output_directory}")
    joblib.dump(model_svr, model_output_directory)

# Function to evaluate the support vector regressor model
from sklearn.metrics import mean_squared_error, r2_score

def evaluation_svr(model_path, test_features_data, test_labels_data, evaluation_output_path):
    """
    This function evaluates the model and saves the evaluation report to the evaluation_output_path.
    """
    model = joblib.load(model_path)

    X_test = pd.read_csv(test_features_data, header=None)
    y_test = pd.read_csv(test_labels_data, header=None)

    y_pred = model.predict(X_test)

    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")
    print(f"R2: {r2_score(y_test, y_pred)};")

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

    with open(evaluation_output_path, "w") as f:
          f.write(json.dumps(report_dict))
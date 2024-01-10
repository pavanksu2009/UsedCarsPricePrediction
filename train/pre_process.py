import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split


# input_data_path for used_cars.csv from local machine 
input_data_path = '../UsedCarsPricePrediction/train/used_cars.csv'
used_cars = pd.read_csv(input_data_path)

target = 'price'
numeric_features = ['Kilometers_Driven', 'Mileage', 'Engine','Power','Seats']
categorical_features = ['Segment']

# X for used_cars.csv from local machine 
X = used_cars.drop(columns=[target])
y = used_cars[target]

# split the data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
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

# train_features_output_path for used_cars.csv from local machine
train_features_output_path = os.path.join("../UsedCarsPricePrediction/train", "train_features.csv")
# train_labels_output_path for used_cars.csv from local machine
train_labels_output_path = os.path.join("../UsedCarsPricePrediction/train", "train_labels.csv")

# validation set for used_cars.csv from local machine
val_features_output_path = os.path.join("../UsedCarsPricePrediction/train", "val_features.csv")
val_labels_output_path = os.path.join("../UsedCarsPricePrediction/train", "val_labels.csv")

# test_features_output_path for used_cars.csv from local machine
test_features_output_path = os.path.join("../UsedCarsPricePrediction/test", "test_features.csv")
# test_labels_output_path for used_cars.csv from local machine
test_labels_output_path = os.path.join("../UsedCarsPricePrediction/test", "test_labels.csv")

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

import os
import argparse # 
import pandas as pd
# import azureml.core
import numpy as np
# import mlflow
# import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from azureml.core import Workspace

def main():
    """Main function of the script."""
 

    # input and output arguments passed by the estimator 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="path to output data")
    args = parser.parse_args()

    ###################
    #<prepare the data>
    ###################
    
    print("input data:", args.data)
    
    data = pd.read_csv(args.data)


    ###################
    #<processing>
    ###################

    # Separate categorical and numerical features
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Apply label encoding to categorical columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Apply data scaling to numerical columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Exporting processed data to local
    processed_data_path = os.path.join(args.output, 'used_cars_processed.csv')
    data.to_csv(processed_data_path, index=False)
    print("processed data is exported to", processed_data_path)

if __name__ == "__main__":
 main()

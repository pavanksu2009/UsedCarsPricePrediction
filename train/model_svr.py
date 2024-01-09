
# import mlflow
import argparse

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

# mlflow.start_run()

def main():
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", type=str, help="path to train data")
    
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
    
        model_svr = SVR()
    
        model_pipeline = make_pipeline(model_svr)
    
        model_pipeline.fit(X_train, y_train)
    
        # print training accuracy and R2 score
        print("training accuracy:", model_pipeline.score(X_train, y_train))
        print("training R2 score:", model_pipeline.score(X_train, y_train))

        # print validation accuracy and R2 score
        print("validation accuracy:", model_pipeline.score(X_val, y_val))
        print("validation R2 score:", model_pipeline.score(X_val, y_val))
        
        # print test accuracy and R2 score
        print("test accuracy:", model_pipeline.score(X_test, y_test))
        print("test R2 score:", model_pipeline.score(X_test, y_test))

if __name__ == '__main__':
    main()

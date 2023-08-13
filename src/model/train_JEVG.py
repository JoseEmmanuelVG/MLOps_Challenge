# Import libraries
import argparse
import glob
import os
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Plot ROC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test, save_path="/workspaces/MLOps_Challenge/src/model/roc_curve.png"):
    # Predictions
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print(f"Accuracy: {acc}")

    # ROC AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print(f"ROC AUC: {auc}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(save_path)
    plt.close()

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model


# define functions
def main(args):
    # Enable autologging
    mlflow.sklearn.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # evaluate model and plot ROC curve
    evaluate_model(model, X_test, y_test)

def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def get_csvs_df(path):
    if path is None:
        raise ValueError("Please provide a valid path to training data using --training_data argument.")
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    
    # Check if the provided path is a directory or a file
    if os.path.isdir(path):
        csv_files = glob.glob(f"{path}/*.csv")
        if not csv_files:
            raise RuntimeError(f"No CSV files found in provided data path: {path}")
        return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    else:
        return pd.read_csv(path)



#def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
#    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str, default='/workspaces/MLOps_Challenge/experimentation/data/diabetes-dev.csv')
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")


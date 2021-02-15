from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.core import Workspace


def clean_dataset(df):
    df1 = df.dropna()
    df1["SEX"] = df1.SEX.apply(lambda x: 0 if x==1 else 1)
    y1 = pd.get_dummies(df1.EDUCATION, prefix="edu")
    df2 = df1.join(y1)
    y2 = pd.get_dummies(df1.MARRIAGE, prefix="marriage")
    df3 = df2.join(y2)
    df3 = df3.drop(columns=["EDUCATION", "MARRIAGE"]).rename(columns={"default payment next month":"Y"})
    return df3


run = Run.get_context()


# +
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--l1', type=float, default=0.5)

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("L1 Ratio (Elastic Net Mixing Parameter):", np.float(args.l1))
    
    ### YOUR CODE HERE ### 
    ws = run.experiment.workspace
    ds = ws.get_default_datastore()
    dataset = Dataset.Tabular.from_delimited_files(path=ds.path("train_data/cleaned_dataset.csv"))
    df = dataset.to_pandas_dataframe()
    y = df["Y"]
    x = df.drop(columns=["Y"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
    
    model = LogisticRegression(penalty = "elasticnet", max_iter=1500, C=args.C, l1_ratio=args.l1, solver="saga", random_state=40).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
# -



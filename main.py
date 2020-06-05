import numpy as np
import pandas as pd
import os, pdb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

def load_dataset(full_path, sensitive_attr):
    dataframe = pd.read_csv(full_path,na_values='?')
    dataframe = dataframe.dropna()
    # split into inputs and outputs
    last_ix = 'income'
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    X, sensitive_attr = dataframe.drop(sensitive_attr, axis=1), dataframe[sensitive_attr]
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    c_ix = []
    n_ix = []
    for i, v in enumerate(dataframe.columns.tolist()):
        if v in cat_ix:
            c_ix.append(i)
        elif v in num_ix:
            n_ix.append(i)
    return X.values, y.values, sensitive_attr, c_ix, n_ix


def run_classifiers( X, y, model, kfold=False):
    if kfold == False:
        perc=0.15
        print(f"Running Train-Test split with {perc}% test set.")
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test,preds)
    
    else:
        k = 10
        print(f"Running {k}k-fold cross validation training.")
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=1)
        # evaluate model
        score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=2)

    return score, X_test, y_test


def demo_parity_check(x_test, y_true, y_pred, sens_attr, accuracy_score):
    # count 1's for men and women 
    count_1 = 0
    count_0 = 0 
    for i, row in enumerate(x_test):
        if sens_attr[i] == 0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 +1 
    print(f"Proportion of 0: {count_0/i}")
    print(f"Proportion of 1: {count_1/i}")


if __name__ == "__main__":
    path_to_file = "../GeneralDatasets/Csv/Adult_NotNA_.csv"
    sensitive_attr = 'sex'
    
    x, y, sens_attr, c_ix, n_ix = load_dataset(path_to_file, sensitive_attr)

    steps = [('c',OneHotEncoder(handle_unknown='ignore'),c_ix), ('n',MinMaxScaler(),n_ix)]
    ct = ColumnTransformer(steps)
    model = GradientBoostingClassifier()
    pipeline = Pipeline(steps=[('t',ct),('m',model)])

    score, x_test, y_test = run_classifiers(x, y, pipeline, False)

    y_pred = model.predict(x_test)

    demo_parity_check(x_test, y_test, y_pred, sens_attr, score)
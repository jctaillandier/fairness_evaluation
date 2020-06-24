import numpy as np
import pandas as pd
import os, pdb, argparse
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
from sklearn.svm import SVC

def load_dataset(full_path, s_attr):
    dataframe = pd.read_csv(full_path,na_values='?')
    dataframe = dataframe.dropna()

    last_ix = 'income'

    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    sensitive_attr =  dataframe[s_attr]
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    c_ix = []
    n_ix = []
    for i, v in enumerate(dataframe.columns.tolist()):
        if v in cat_ix:
            c_ix.append(i)
        elif v in num_ix:
            n_ix.append(i)
    return X.values, y.values, c_ix, n_ix, sensitive_attr


def run_classifiers(X, y, model):
    perc=0.10
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=perc, random_state=76)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test,preds)

    return score, X_test, y_test

# sex = 1 means male
def demo_parity_check(x_test, y_true, y_pred, sens_attr, accuracy_score):
    '''
        Calculates demographic parity on trained model.

        The proportion of 1 (where decision is positive) should be similar for both sex 
    '''
    count_1, count_1_men, count_1_fem, count_0_men, count_0_fem, count_0 = 0,0,0,0,0,0
    incr = lambda x: x+1 

    for i, row in enumerate(x_test):
        if sens_attr[i] == 0:
            count_1 = incr(count_1)
            if row[-1] == 0: #female
                count_1_men = incr(count_1_men)
            else:
                count_1_fem = incr(count_1_fem)
        else:
            count_0 = incr(count_0) 
            if row[-1] == 0: #female
                count_0_men = incr(count_0_men)
            else:
                count_0_fem = incr(count_0_fem)
    
    print(f"Proportion of 1 given sex=male: {count_1_men/i:.3f}")
    print(f"Proportion of 1 given sex=female: {count_1_fem/i:.3f}")
    print(f"proportion of label 1: {count_1/i:.3f}")

def disparate_impact(x_test, y_true, y_pred, sens_attr, accuracy_score):
    '''
        Calculates disparate impact. 

        Output should be greater than 0.8 to meet legal definition of disparate impact, but the closest to 1 the better
    '''
    count_1, count_1_men, count_1_fem, count_0_men, count_0_fem, count_0 = 0,0,0,0,0,0
    incr = lambda x: x+1 

    for i, row in enumerate(x_test):
        if sens_attr[i] == 0:
            count_1 = incr(count_1) # count of total positive decision
            if row[-1] == 0: # female
                count_1_men = incr(count_1_men)
            else:
                count_1_fem = incr(count_1_fem)
        else:
            count_0 = incr(count_0) # count of total negative decision
            if row[-1] == 0: # female
                count_0_men = incr(count_0_men)
            else:
                count_0_fem = incr(count_0_fem)
    # prob oui if men / prob oui if female
    a = count_1_men/(count_1_men+count_0_men)
    b = count_1_fem/(count_1_fem+count_0_fem)
    print(f"Disparate Impact value: {a/b:.3f}")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, default='tree', help='Type of model', choices=['svm', 'tree', 'MLP','bagging', 'grad_boost', 'forest'], required=False)
    parser.add_argument('-fn', '--file_name', default="disp_impact_remover_1.0_sex.csv" , type=str, help='Path to the file to use as input.')
    parser.add_argument('-s', '--sensitive', type=str, default='sex', help='attribute considered sensitive for calculations')
    parser.add_argument('-enc', '--encode', type=str, default='true', help='Whether to encode features if not already encoded input', required=False)
    
    return parser.parse_args()

model_dict = {
    'tree': DecisionTreeClassifier(),
    'forest': RandomForestClassifier(),
    'grad_boost': GradientBoostingClassifier(),
    'bagging': BaggingClassifier(),
    'MLP': MLPClassifier(),
    'svm': SVC()
}

if __name__ == "__main__":
    args = get_args()

    path_to_file = "../GeneralDatasets/sex_last/Adult_NotNA__sex.csv"
    path_to_file = f"./fairness_calculation_data/{args.file_name}"
    
    x, y, c_ix, n_ix, sensitive_col = load_dataset(path_to_file, args.sensitive)
    
    model = model_dict[args.model]

    if args.encode=='true':
        print('Encoding. \n')
        steps = [('c',OneHotEncoder(handle_unknown='ignore'),c_ix), ('n',MinMaxScaler(),n_ix)]
        ct = ColumnTransformer(steps)
        pipeline = Pipeline(steps=[('t',ct),('m',model)])
        score, x_test, y_test = run_classifiers(x, y, pipeline)
    else:
        print('No Encoding. \n')
        score, x_test, y_test = run_classifiers(x, y, model)
    print(f"Accuracy on decision: {(score*100):.3f}% \n")

    y_pred = model.predict(x_test)
    
    demo_parity_check(x_test, y_test, y_pred, sensitive_col, score)

import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import X_train, train_targets, X_val, val_targets

model = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=256, random_state=42)

model.fit(X_train, train_targets)

if __name__ == "__main__":

    # train_preds = model.predict(X_train)
    # train_probs = model.predict_proba(X_train)
    # print(train_preds)
    # print(train_probs)
    # print(accuracy_score(train_targets, train_preds))
    print(model.score(X_train, train_targets))
    print(model.score(X_val, val_targets))

    # value_c = pd.Series(train_preds).value_counts()
    # print(value_c)

    print('-------------------------')
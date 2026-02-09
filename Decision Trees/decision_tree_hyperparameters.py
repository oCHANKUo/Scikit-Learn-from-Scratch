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

## Experiment with Depth
def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, train_targets)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
    return {'Max Depth' : md, 'Training Error' : train_acc, 'Validation Error' : val_acc}

errors_df = pd.DataFrame([max_depth_error(md) for md in range(1,21)])

if __name__ == "__main__":

    '''
    plt.figure()
    plt.plot(errors_df['Max Depth'], errors_df['Training Error'])
    plt.plot(errors_df['Max Depth'], errors_df['Validation Error'])
    plt.title('Training vs. Validation Error')
    plt.xticks(range(0,21,2))
    plt.xlabel('Max. Depth')
    plt.ylabel('Prediction Error (1 - Accuracy)')
    plt.legend(['Training', 'Validation'])
    plt.show()
    '''

    ## 7 is the optimal depth
    model = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train, train_targets)
    print(model.score(X_val, val_targets))

    print('-------------------------')
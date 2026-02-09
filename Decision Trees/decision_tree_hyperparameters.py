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
def max_depth_error(md, mln):
    model = DecisionTreeClassifier(max_depth=md, random_state=42, max_leaf_nodes=mln)
    model.fit(X_train, train_targets)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
    return {'Max Depth' : md, 'Max Leaf Nodes' : mln, 'Training Error' : train_acc, 'Validation Error' : val_acc}

max_leaf_nodes_range = [2, 4, 8, 16, 32, 64, 128, 256]
max_depth_range = range(1, 21)

errors = []

for md in max_depth_range:
    for mln in max_leaf_nodes_range:
        model = DecisionTreeClassifier(max_depth=md, max_leaf_nodes=mln,random_state=42)
        model.fit(X_train, train_targets)

        val_error = 1 - model.score(X_val, val_targets)

        errors.append({
            'Max Depth': md,
            'Max Leaf Nodes': mln,
            'Validation Error': val_error
        })

errors_df = pd.DataFrame(errors)

if __name__ == "__main__":

    heatmap_data = errors_df.pivot(index='Max Leaf Nodes', columns='Max Depth', values='Validation Error')

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis_r', cbar_kws={'label': 'Validation Error'})
    plt.title('Validation Error Heatmap (Depth vs Max Leaf Nodes)')
    plt.xlabel('Max Depth')
    plt.ylabel('Max Leaf Nodes')
    plt.tight_layout()
    plt.show()

    best_params = errors_df.loc[errors_df['Validation Error'].idxmin()]
    print(best_params)

    ## 7 is the optimal depth
    # model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42).fit(X_train, train_targets)
    # print(model.score(X_train, train_targets))
    # print(model.score(X_val, val_targets))
    # print(model.tree_.max_depth)

    print('-------------------------')
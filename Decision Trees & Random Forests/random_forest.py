import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
from preprocessing import X_train, train_targets, X_val, val_targets

model = RandomForestClassifier(n_jobs=-1, random_state=42)

model.fit(X_train, train_targets)

if __name__ == "__main__":

    # print(model.score(X_train, train_targets))
    # print(model.score(X_val, val_targets))

    # Probablity
    # print(model.predict_proba(X_train))

    # plt.figure(figsize=(16,4))
    # plot_tree(model.estimators_[0], max_depth=2, feature_names=X_train.columns, filled=True, rounded=True, class_names=model.classes_)
    # plt.figure(figsize=(16,4))
    # plot_tree(model.estimators_[15], max_depth=2, feature_names=X_train.columns, filled=True, rounded=True, class_names=model.classes_)
    
    plt.show()

    # Importance
    importance_df = pd.DataFrame({'feature': X_train.columns,
                                  'importance': model.feature_importances_
                                  }).sort_values('importance', ascending=False)
    print(importance_df.head(10))

    print("---------------------------")


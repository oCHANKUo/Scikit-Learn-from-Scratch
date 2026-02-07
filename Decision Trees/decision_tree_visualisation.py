import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import X_train, train_targets, X_val, val_targets
from decision_tree import model

if __name__ == "__main__":

    plt.figure(figsize=(16, 4))
    plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)
    plt.show()

    print('-------------------------')
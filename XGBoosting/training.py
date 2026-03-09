import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import seaborn as sns

from preprocessing import X, targets
from data_prep import merged_df, merged_test_df

rcParams['figure.figsize'] = 15,15
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')

model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)

if __name__ == "__main__":

    # train
    model.fit(X, targets)

    # predict
    preds = model.predict(X)
    print(preds)

    # Evaluation
    print("rmse: ", root_mean_squared_error(preds, targets))

    # Visualization
    # plot_tree(model, tree_idx=0, rankdir='LR')
    # plot_tree(model, tree_idx=0, rankdir='LR', num_trees=1)
    # plot_tree(model, tree_idx=0, rankdir='LR', num_trees=19)

    # Visualize trees as text
    trees = model.get_booster().get_dump()
    print(len(trees))

    # Feature Importance
    ''' Feature Importance '''
    importance_df = pd.DataFrame({
        'feature' : X.columns,
        'importance' : model.feature_importances_
    }).sort_values('importance',  ascending=False)
    print(importance_df.head(10))

    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.show()
   
    print("--------------------")
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
import numpy as np

from preprocessing import X, targets, X_test
from data_prep import merged_df, merged_test_df

rcParams['figure.figsize'] = 15,15
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')

# model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)
model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=1000, 
                     learning_rate=0.2, max_depth=10, subsample=0.9, 
                     colsample_bytree=0.7)

X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size=0.1)

# K fold evaluation
def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = root_mean_squared_error(model.predict(X_train), train_targets)
    val_rmse = root_mean_squared_error(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse

def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)

''' Hyperparameter Testing and K Fold Cross Validation '''
def test_params_kfold(n_splits, **params):
    train_rmse, val_rmse, models = [], [], []
    kfold = KFold(n_splits)
    for train_idxs, val_idxs in kfold.split(X):
        X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
        X_val, val_targets = X.iloc[val_idxs], targets.iloc(X_train, train_targets, X_val, val_targets, **params)
        models.append(model)
        train_rmse.append(train_rmse)
        val_rmse.append(val_rmse)
    print('Train RMSE: {}, Validation RMSE: {}' .format(np.mean(train_rmse), np.mean(val_rmse)))
    return models

def test_params(**params):
    model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_rmse = root_mean_squared_error(model.predict(X_train), train_targets)
    val_rmse = root_mean_squared_error(model.predict(X_val), val_targets)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))


kfold = KFold(n_splits=5)

models = []

if __name__ == "__main__":

    # train
    # model.fit(X, targets)

    # Evaluation
    # print("rmse: ", root_mean_squared_error(preds, targets))

    # Visualization
    # plot_tree(model, tree_idx=0, rankdir='LR')
    # plot_tree(model, tree_idx=0, rankdir='LR', num_trees=1)
    # plot_tree(model, tree_idx=0, rankdir='LR', num_trees=19)

    # Visualize trees as text
    # trees = model.get_booster().get_dump()
    # print(len(trees))

    # Feature Importance
    ''' Feature Importance '''
    '''importance_df = pd.DataFrame({
        'feature' : X.columns,
        'importance' : model.feature_importances_
    }).sort_values('importance',  ascending=False)'''
    # print(importance_df.head(10))

    # sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    # plt.show()


    # Evaluation
    for train_idxs, val_idxs in kfold.split(X):
        X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
        X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
        model, train_rmse, val_rmse = train_and_evaluate(X_train, train_targets, X_val, val_targets, max_depth=4, n_estimators=20)
        models.append(model)
        # print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))

    # predict
    # preds = predict_avg(models, X)
    # print(preds)

    ''' n_estimators '''
    # test_params(n_estimators=240)
    # test_params(max_depth=10)
    # test_params(n_estimators=50, learning_rate=0.99)
    # test_params(booster='gblinear')

    ''' max depth '''
    # test_params(max_depth=5)
    # test_params(max_depth=10)

    ''' Learning Rate: Scaling factor applied to the prediction of each tree. High learning rate = overfitting '''
    # test_params(n_estimators=50, learning_rate=0.01)
    # test_params(n_estimators=50, learning_rate=0.1)
    # test_params(n_estimators=50, learning_rate=0.99)

    ''' Booster: Instead of decision trees, we can also train a linear model (even though its not well suited) '''
    # test_params(booster='gblinear')


    ''' Final '''
    # model.fit(X, targets)
    # print(model.predict(X_test))
    test_params(n_estimators=1000, 
                     learning_rate=0.2, max_depth=10, subsample=0.9, 
                     colsample_bytree=0.7)
   
    print("--------------------")
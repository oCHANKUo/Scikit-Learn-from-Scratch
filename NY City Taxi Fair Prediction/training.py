import opendatasets as od
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

from data_prep import df, test_df
from preprocessing import train_inputs, train_targets, val_inputs, val_targets

class MeanRegressor():
    def fit(self, inputs, targets):
        self.mean = targets.mean()

    def predict(self, inputs):
        return np.full(inputs.shape[0], self.mean)

mean_model = MeanRegressor()
mean_model.fit(train_inputs, train_targets)

linreg_model = LinearRegression()
linreg_model.fit(train_inputs, train_targets)

if __name__ == "__main__":

    # print(mean_model.mean)

    ''' Mean Model '''
    # train_preds = mean_model.predict(train_inputs)
    # val_preds = mean_model.predict(val_inputs)

    ''' Linear Regression Model '''
    train_preds = linreg_model.predict(train_inputs)
    val_preds = linreg_model.predict(val_inputs)
    print(train_preds)
    print(val_preds)

    train_rmse = root_mean_squared_error(train_targets, train_preds)
    val_rmse = root_mean_squared_error(val_targets, val_preds)
    print(train_rmse)
    print(val_rmse)

    print("--------------")
import os
import opendatasets as od
import pandas as pd
from data_prep import merged_df, merged_test_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from preprocessing import X, targets
from training import model

rcParams['figure.figsize'] = 30,30

if __name__ == "__main__":

    plot_tree(model, rankdir='LR')
   
    print("--------------------")
import opendatasets as od
import os
import pandas as pd
import numpy as np
from data_preprocess import merged_df, merged_test_df
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

train_size = int(.75 * len(merged_df))

sorted_df = merged_df.sort_values('Date')
train_df, val_df = sorted_df[:train_size], sorted_df[train_size:]

input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Day', 'Month', 'Year']
target_col = 'Sales'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = merged_test_df[input_cols].copy()

numeric_cols = ['Store', 'Day', 'Month', 'Year']
categorical_cols = ['DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment']

'''Imputation'''
imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

'''Scaling'''
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

'''Encoding'''
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

'''Extract out numeric data'''
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

'''Function to return a mean value'''
def return_mean(inputs):
    return np.full(len(inputs), merged_df.Sales.mean())

def guess_random(inputs):
    lo, hi = merged_df.Sales.min(), merged_df.Sales.max()
    return np.random.random(len(inputs)) * (hi - lo) + lo

train_preds1 = return_mean(X_train)
train_preds2 = guess_random(X_train)

'''Baseline Model'''
linreg = LinearRegression()
linreg.fit(X_train, train_targets)
train_preds = linreg.predict(X_train)

val_preds = linreg.predict(X_val)

if __name__ == "__main__":

    # print(len(merged_df))
    # print(len(train_df), len(val_df))
    # print(train_df.columns)

    '''Root Mean Squared Error'''
    # print(root_mean_squared_error(train_preds1, train_targets))
    # print(root_mean_squared_error(return_mean(X_val), val_targets)) # Helps check the error

    # print(root_mean_squared_error(train_preds2, train_targets))
    # print(root_mean_squared_error(guess_random(X_val), val_targets))

    print(root_mean_squared_error(train_preds, train_targets))
    print(root_mean_squared_error(val_preds, val_targets))
    print('------------------')
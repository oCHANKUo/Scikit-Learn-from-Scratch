import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'

raw_df = pd.read_csv(train_csv)
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
# train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

## Identify categorical and Numerical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols =  train_inputs.select_dtypes('object').columns.tolist()

## Imputation
imputer = SimpleImputer(strategy= 'mean')
imputer.fit(raw_df[numeric_cols])
## Overwrite the numeric columns of the data columns
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

## Scaling
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


if __name__ == "__main__":

    

    # print('test_df.shape: ', test_df.shape)

    # print(list(imputer.statistics_))

    # print(train_inputs[numeric_cols].describe())

    # Check missing values
    # print(train_inputs[numeric_cols].isna().sum())

    print('------------------------------')

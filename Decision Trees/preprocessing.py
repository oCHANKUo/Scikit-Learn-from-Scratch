import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'

raw_df = pd.read_csv(train_csv)

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

# Splitting data into train, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

# Input and Target Split
input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()
val_inputs = train_df[input_cols].copy()
val_targets = train_df[target_cols].copy()
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()

if __name__ == "__main__":

    # raw_df.info()

    print('train_df.shape: ', train_df.shape)
    print('val_df.shape: ', val_df.shape)
    print('test_df.shape: ', test_df.shape)


    print('-------------------------')
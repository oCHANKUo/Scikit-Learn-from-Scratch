import opendatasets as od
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from data_prep import df, test_df

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

''' Fill/Remove Missing Values '''
train_df = train_df.dropna()
val_df = val_df.dropna()

''' Extract Inputs and Outputs '''
input_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target_col = 'fare_amount'

''' Training '''
train_inputs = train_df[input_cols]
train_targets = train_df[target_col]

''' Validation '''
val_inputs = val_df[input_cols]
val_targets = val_df[target_col]

''' Test '''
test_inpts = test_df[input_cols]

if __name__ == "__main__":

    print(len(train_df), len(val_df))

    print("--------------")
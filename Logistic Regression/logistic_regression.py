import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":

    print('train_df.shape: ', train_df.shape)
    print('val_df.shape: ', val_df.shape)
    print('test_df.shape: ', test_df.shape)

    print('------------------------------')

import opendatasets as od # Easier way to download Kaggle datasets
import os
import pandas as pd

dataset_url = 'https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package'
# od.download(dataset_url)

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'

raw_df = pd.read_csv(train_csv)
# Drop rows where RainToday or RainTomorrow are null
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

if __name__ == "__main__":
    print(raw_df.info())
    print('-----------------')
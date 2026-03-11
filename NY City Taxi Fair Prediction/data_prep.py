import opendatasets as od
import pandas as pd
import random

dataset_url = 'https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data'

# od.download(dataset_url)

data_dir = 'D:/new-york-city-taxi-fare-prediction'

sample_frac = 0.01

selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'passenger_count': 'float32'
}

def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_frac

random.seed(42)
df = pd.read_csv(data_dir+"/train.csv",
                 usecols=selected_cols,
                 dtype=dtypes,
                 parse_dates=['pickup_datetime'],
                 skiprows=skip_row, 
                 encoding="latin1")

test_df = pd.read_csv(data_dir+'/test.csv', 
                      dtype=dtypes, 
                      parse_dates=['pickup_datetime'])

if __name__ == "__main__":

    # print(df)
    # print(test_df)

    # print(df.info())
    # print(df.describe())

    # print(test_df.info())
    # print(test_df.describe())

    print("--------------")
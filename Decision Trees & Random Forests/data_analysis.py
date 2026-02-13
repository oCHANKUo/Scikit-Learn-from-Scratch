import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

dataset_url = 'https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package'
# od.download(dataset_url)

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'
os.listdir(data_dir)

raw_df = pd.read_csv(train_csv)

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

if __name__ == "__main__":

    raw_df.info()


    print('-------------------------')
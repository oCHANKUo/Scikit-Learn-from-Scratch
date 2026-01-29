import opendatasets as od # Easier way to download Kaggle datasets
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

raw_df = pd.read_csv(train_csv)

# Sampling step is not necessary but can be valuable with large datasets
use_sample = False
sample_fraction = 0.1
if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()

# Drop rows where RainToday or RainTomorrow are null
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

if __name__ == "__main__":
    # print(raw_df.info())

    fig = px.histogram(raw_df, 
                       x='Location', 
                       title='Location vs. RainyDays', 
                       color='RainToday')
    
    fig2 = px.histogram(raw_df, 
                        x='Temp3pm',
                        title='Temperature 3pm vs Rain Tomorrow',
                        color='RainTomorrow')
    
    fig3 = px.histogram(raw_df,
                        x='RainTomorrow',
                        color='RainToday',
                        title='Rain Tomorrow vs Rain Today')
    
    fig4 = px.scatter(raw_df.sample(2000),
                      title='Min Temp vs Max Temp',
                      x='MinTemp',
                      y='MaxTemp',
                      color='RainToday')
    
    fig5 = px.strip(raw_df.sample(2000),
                    title='Temp (3pm )vs Humidity (3pm)',
                    x='Temp3pm',
                    y='Humidity3pm',
                    color='RainTomorrow')
    fig5.show()
    print('-----------------')
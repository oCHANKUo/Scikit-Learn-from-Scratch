import opendatasets as od
import os

dataset_url = 'https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package'
od.download(dataset_url)

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'
os.listdir(data_dir)

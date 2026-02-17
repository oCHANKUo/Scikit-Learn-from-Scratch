import opendatasets as od


dataset_url = 'https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package'
od.download(dataset_url)

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'
os.listdir(data_dir)

raw_df = pd.read_csv(train_csv)
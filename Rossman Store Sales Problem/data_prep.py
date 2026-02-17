import opendatasets as od
import os
import pandas as pd

od.download('https://www.kaggle.com/c/rossmann-store-sales')

if __name__ == "__main__":

    os.listdir('rossmann-store-sales')
    # ross_df = pd.read_csv('./rossmann-store-sales/train.csv', low_memory=False)

    print('------------------')
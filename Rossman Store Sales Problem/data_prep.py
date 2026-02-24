import opendatasets as od
import os
import pandas as pd

# od.download('https://www.kaggle.com/c/rossmann-store-sales')

ross_df = pd.read_csv('./rossmann-store-sales/train.csv', low_memory=False)
store_df = pd.read_csv('./rossmann-store-sales/store.csv')
test_df = pd.read_csv('./rossmann-store-sales/test.csv')

# Merge ross_df and store_df to get a richer set of features for each row of the training set
merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

if __name__ == "__main__":

    # print(os.listdir('rossmann-store-sales'))

    # print(merged_df.shape)
    # print(merged_test_df)

    print('------------------')
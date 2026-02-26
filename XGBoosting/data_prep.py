import os
import opendatasets as od
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)

# od.download('https://www.kaggle.com/c/rossmann-store-sales')

ross_df = pd.read_csv('./rossmann-store-sales/train.csv', low_memory=False)
store_df = pd.read_csv('./rossmann-store-sales/store.csv')
test_df = pd.read_csv('./rossmann-store-sales/test.csv')
submission_df = pd.read_csv('./rossmann-store-sales/sample_submission.csv')

merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

if __name__ == "__main__":

    print(ross_df)

    print("--------------------")
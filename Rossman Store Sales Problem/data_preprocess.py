import opendatasets as od
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_prep import ross_df, merged_df, merged_test_df

merged_df['Date'] = pd.to_datetime(merged_df.Date)
merged_test_df['Date'] = pd.to_datetime(merged_test_df.Date)

'''Exclude the dates when the store was closed'''
merged_df = merged_df[merged_df.Open==1].copy()

'''Feature Engineering'''
merged_df['Day'] = merged_df.Date.dt.day
merged_df['Month'] = merged_df.Date.dt.month
merged_df['Year'] = merged_df.Date.dt.year

merged_test_df['Day'] = merged_test_df.Date.dt.day
merged_test_df['Month'] = merged_test_df.Date.dt.month
merged_test_df['Year'] = merged_test_df.Date.dt.year

if __name__ == "__main__":

    # print(merged_df.info())
    # print(round(merged_df.describe().T,2))
    # print(merged_df.duplicated().sum())
    # print(merged_df.Date.min(), merged_df.Date.max())
    # print(merged_test_df.Date.min(), merged_test_df.Date.max())

    '''Distribution of the target "Sales" column'''
    # print(sns.histplot(data=merged_df, x='Sales'))
    # plt.show()

    '''There is a bunch of 0 values. Check if its because the store was closed'''
    # print(merged_df.Open.value_counts())
    # print(merged_df.Sales.value_counts()[0])

    '''Explore other columns such as, Sales vs. Customers using scatter plot, Stores vs Sales, Day of the Week vs Sales, Promo vs Sales'''

    '''After Feature Engineering'''
    # sns.barplot(data=merged_df, x='Year', y='Sales')
    # sns.barplot(data=merged_df, x='Month', y='Sales')

    print('------------------')
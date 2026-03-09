import os
import opendatasets as od
import pandas as pd
from data_prep import merged_df, merged_test_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

'''Convert Date to datecolumn to extract different parts of the data'''
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['WeekOfYear'] = df.Date.dt.isocalendar().week
    df['Day'] = df.Date.dt.day

split_date(merged_df)
split_date(merged_test_df)


'''Sales are 0 whenever the store is closed. Remove the rows where store is closed'''
merged_df = merged_df[merged_df.Open == 1].copy()


'''Use columns CompetitionOpenSince[Month/Year] from store_df to compute the number of months its been open'''
def comp_months(df):
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x < 0 else x).fillna(0)

'''Additional Columns to indicate how long a store has been running Promo2 and whether a new round of Promo2 starts in the current month'''
def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    try:
        months = (row['PromoInterval'] or '').split(',')
        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

def promo_cols(df):
    # Months since the Promo2 was open
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) * 7/30.5
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']
    #Whether a enw round of promotions was started in the current month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']

comp_months(merged_df)
comp_months(merged_test_df)
promo_cols(merged_df)
promo_cols(merged_test_df)

'''Input and Target Columns'''
input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
              'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen', 
              'Day', 'Month', 'Year', 'WeekOfYear',  'Promo2', 
              'Promo2Open', 'IsPromo2Month']
target_col = 'Sales'

'''Numeric and Categorical Cols'''
numeric_cols = ['Store', 'Promo', 'SchoolHoliday', 
              'CompetitionDistance', 'CompetitionOpen', 'Promo2', 'Promo2Open', 'IsPromo2Month',
              'Day', 'Month', 'Year', 'WeekOfYear',  ]
categorical_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']


'''Input and Target Columns'''
inputs = merged_df[input_cols].copy()
inputs[numeric_cols].isna().sum()
targets = merged_df[target_col].copy()
test_inputs = merged_test_df[input_cols].copy()
test_inputs[numeric_cols].isna().sum()


''' competition distance is the only missing value, and we can simply fill it with the highest value (to indicate that competition is very far away).'''
max_distance = inputs.CompetitionDistance.max()

inputs['CompetitionDistance'] = inputs['CompetitionDistance'].fillna(max_distance)
test_inputs['CompetitionDistance'] = test_inputs['CompetitionDistance'].fillna(max_distance)

''' Scaling '''
scaler = MinMaxScaler().fit(inputs[numeric_cols])

inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

''' Encoding '''
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

''' Extracting Numeric Data for Training (No validation set  since we are using K Fold cross validation)'''
X = inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


if __name__ == "__main__":

    # print(merged_df[merged_df.Open == 0].Sales.value_counts())
    # print(merged_df)

    # print(merged_df[['Date', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Promo2Open', 'IsPromo2Month']].sample(20))
    

    
    print("--------------------")
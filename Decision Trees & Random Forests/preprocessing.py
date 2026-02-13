import opendatasets as od 
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

data_dir = 'weather-dataset-rattle-package'
train_csv = data_dir + '/weatherAUS.csv'

raw_df = pd.read_csv(train_csv)

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

# Splitting data into train, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

# Input and Target Split
input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_cols].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy()

# Numeric and Categorical Columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include='string').columns.tolist()

## Imputing
imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scaling
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_encoded = pd.DataFrame(
    encoder.transform(train_inputs[categorical_cols]),
    columns=encoded_cols,
    index=train_inputs.index
)
train_inputs = pd.concat([train_inputs, train_encoded], axis=1)

val_encoded = pd.DataFrame(
    encoder.transform(val_inputs[categorical_cols]),
    columns=encoded_cols,
    index=val_inputs.index
)
val_inputs = pd.concat([val_inputs, val_encoded], axis=1)

test_encoded = pd.DataFrame(
    encoder.transform(test_inputs[categorical_cols]),
    columns=encoded_cols,
    index=test_inputs.index
)
test_inputs = pd.concat([test_inputs, test_encoded], axis=1)

# Drop Textual Categorical Data
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

if __name__ == "__main__":

    # raw_df.info()

    # print('train_df.shape: ', train_df.shape)
    # print('val_df.shape: ', val_df.shape)
    # print('test_df.shape: ', test_df.shape)

    # print(numeric_cols)
    # print(categorical_cols)

    # print(test_inputs[numeric_cols].isna().sum()) # Check missing values

    # print(val_inputs.describe().loc[['min', 'max']]) # Check min mac values for each columns. Should be between 0 and 1
    
    print(X_test)

    print('-------------------------') 
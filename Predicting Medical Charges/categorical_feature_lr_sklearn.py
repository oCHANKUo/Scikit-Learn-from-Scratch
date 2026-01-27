'''
We only used Numerical columns linear regressional before, since computations need to be done with numbers.
We can fit a model for the entire dataset if we can also use categorical features.

To use categorical features, we need to convert them into numbers.
there are 3 common techniques.
1. We can use 0, 1 for two-category columns or binary categories.
2. If it has 2< categories, we can perform one-hot encoding. i.e. Create a new column for each category with 1s and 0s.
3. If the categories have natural order, they can be converted into numbers, 1, 2, 3, 4... Called Ordinals.
'''

from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import plot_graph
from linear_regression import try_parameter, rmse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing

medical_df = pd.read_csv("medical.csv")

# Binary Categories.
'''Smoker column can be converted to 0 and 1 since it only has, yes or no'''
smoker_codes = {'no' : 0, 'yes' : 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

'''Similarly, Sex column can also be converted. But Sex does not have a noiceable impact on the result.'''
sex_codes = {'female' : 0, 'male' : 1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)

# One-Hot Encoding
'''Region column has 4 values and no order. We can use one-hot encoding'''
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
# print(enc.categories_)

one_hot = enc.transform(medical_df[['region']]).toarray()
# print(one_hot)

medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot # this will create 4 new columns, and assign the one-shot values taken from region


'''
charges = w1 * age + w2 + bmi + w3 * children + w4 * smoker + w5 * sex + w6 * region + b
'''
input_cols = ['age', 'bmi', 'children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
inputs, targets = medical_df[input_cols], medical_df.charges

model = LinearRegression()
model.fit(inputs, targets)

predictions = model.predict(inputs)
loss = rmse(targets, predictions)


if __name__ == "__main__":
    # sns.barplot(data=medical_df, x='smoker', y='charges')
    # print(medical_df)
    # print(medical_df.charges.corr(medical_df.smoker_code))
    print(predictions)
    print(loss)

    # sns.barplot(data=medical_df, x='sex', y='charges')

    print('-----')


'''
The extra categorical features did not do much of a difference. Maybe we can use 2 seperate models for predictions. 
One for Smokers
The other for non-smokers
with the same features.
'''


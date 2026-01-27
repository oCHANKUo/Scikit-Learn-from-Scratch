
'''
standardization = (value - mean)/standard deviation
Standardization ensures all features are on a similar scale.
This helps the model learn faster and prevents features with large values
from dominating the learning process, especially for gradient-based methods.
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
from sklearn.preprocessing import StandardScaler

medical_df = pd.read_csv("medical.csv")

# normal categorisation as before
smoker_codes = {'no' : 0, 'yes' : 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
sex_codes = {'female' : 0, 'male' : 1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
one_hot = enc.transform(medical_df[['region']]).toarray()
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot


numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])

# scale the data
scaled_inputs = scaler.transform(medical_df[numeric_cols])

cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values

inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges

model = LinearRegression()
model.fit(inputs, targets)

predictions = model.predict(inputs)
loss = rmse(targets, predictions)

if __name__ == "__main__":
    # print(scaler.mean_) # mean
    # print(scaler.var_) # variance/standard deviation
    # print(scaled_inputs)
    print(predictions)
    print(loss)


    print('----------------')


'''
Scaling does not effect the Loss. Instead it changes the Weights.
When testing for values, they need to be scaled as well
'''


'''
Finally, How to approach a machine learning problem
1. Explore the data and find correlations between input - target
2. Pick the right model, loss functions and optimzer
3. Scale numeric variables and one hot encode categorical data
4. Set aside a test set
5. Train model
6. Make predictions on the test set
'''
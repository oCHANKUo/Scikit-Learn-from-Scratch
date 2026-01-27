from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import plot_graph
from linear_regression import try_parameter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

medical_df = pd.read_csv("medical.csv")
non_smoker_df = medical_df[medical_df.smoker == 'no']
smoker_df = medical_df[medical_df.smoker == 'yes']

# Initializes a Linear Regression model
model = LinearRegression()
model2 = SGDRegressor()

inputs = non_smoker_df[['age']] # 2 [[]], because model.fit requires 2 dimensional arrays
smoker_inputs = smoker_df[['age']]
targets = non_smoker_df.charges # what i want to predict
smoker_targets = smoker_df.charges

# trains the model
# model.fit(inputs, targets) 
model.fit(smoker_inputs, smoker_targets)

# predictions = model.predict(np.array([[23],
#                                       [37],
#                                       [61]]))

predictions = model.predict(smoker_inputs)

# w
w = model.coef_

# b
b = model.intercept_

if __name__ == "__main__":
    # print(predictions)
    print(w)
    print(b)
    try_parameter(w, b, smoker_df)
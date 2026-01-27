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

# Initializes a Linear Regression model
model = LinearRegression()
model2 = SGDRegressor()

inputs = non_smoker_df[['age']] # 2 [[]], because model.fit requires 2 dimensional arrays
targets = non_smoker_df.charges # what i want to predict

# trains the model
# model.fit(inputs, targets) 
model2.fit(inputs, targets)

# predictions = model.predict(np.array([[23],
#                                       [37],
#                                       [61]]))

predictions = model2.predict(inputs)

# w
w = model2.coef_

# b
b = model2.intercept_

if __name__ == "__main__":
    print(predictions)
    print(w)
    print(b)
    try_parameter(w, b)
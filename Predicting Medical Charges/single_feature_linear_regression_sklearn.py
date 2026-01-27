from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import plot_graph
from sklearn.linear_model import LinearRegression

medical_df = pd.read_csv("medical.csv")
non_smoker_df = medical_df[medical_df.smoker == 'no']

# Initializes a Linear Regression model
model = LinearRegression()

inputs = non_smoker_df[['age']] # 2 [[]], because model.fit requires 2 dimensional arrays
targets = non_smoker_df.charges # what i want to predict

model.fit(inputs, targets) # trains the model

# predictions = model.predict(np.array([[23],
#                                       [37],
#                                       [61]]))

# predictions = model.predict(inputs)

# w
w = model.coef_

# b
b = model.intercept_

if __name__ == "__main__":
    print(w)
    print(b)
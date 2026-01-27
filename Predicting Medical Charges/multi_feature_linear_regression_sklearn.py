# Adding another feature is fairly straightforward
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

medical_df = pd.read_csv("medical.csv")
non_smoker_df = medical_df[medical_df.smoker == 'no']
smoker_df = medical_df[medical_df.smoker == 'yes']

# Create inputs and targets
inputs = non_smoker_df[['age', 'bmi']] 
smoker_inputs = smoker_df[['age']]
triple_input = non_smoker_df[['age', 'bmi', 'children']]
total_dataset_input = medical_df[['age', 'bmi', 'children']]


targets = non_smoker_df.charges
total_dataset_targets = medical_df.charges
smoker_targets = smoker_df.charges

# Create and train the model
model = LinearRegression().fit(total_dataset_input, total_dataset_targets)

if __name__ == "__main__":

    # Predict
    predictions = model.predict(total_dataset_input)

    print("predictions =", predictions)
    print("loss = ", rmse(total_dataset_targets, predictions))

'''
This is still a Linear Relationship. 
But it becomes a 3 dimensional relationship, now that BMI is also considered.
charges = w1 * age + w2 * bmi + b
BMI doesnt reducse the loss by much because BMI has very weak correlation with charges
'''
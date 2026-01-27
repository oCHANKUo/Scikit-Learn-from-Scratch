
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

medical_df = pd.read_csv("medical.csv")
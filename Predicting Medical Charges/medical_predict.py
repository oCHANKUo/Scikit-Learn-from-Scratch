# Linear Regression
from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Import the dataset and save as a CSV.
medical_charges_url = "https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
urlretrieve(medical_charges_url, 'medical.csv')

# Step 2: Create a Pandas dataframe to view and analyse the data
medical_df = pd.read_csv('medical.csv')
# print(medical_df)

# medical_df.info() # Check the data types
stat_info = medical_df.describe() # Statistical summary
print(stat_info) 
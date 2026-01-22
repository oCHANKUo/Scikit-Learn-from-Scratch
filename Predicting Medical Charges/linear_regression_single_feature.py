from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import plot_graph


medical_df = pd.read_csv("medical.csv")

''' Smoker and Age columns have the strongest correlation with charges
# But Smoker value cannot be used since its simply "Yes" or "No" '''

non_smoker_df = medical_df[medical_df.smoker == 'no']

# plot_graph('age_vs_charges') # theres a generally linear relationship between age and charges

''' 
Charges = W * age + b.
w = wights in machine learning or Slope in stats.
b = bias in ML or intercept in Stats
Age = inputs
Charges = target
'''

# Simple linear relation
def estimate_charges(age, w, b):
    return w * age + b

# Compare our manual estimation with actual relation
def try_parameter(w, b):
    ages = non_smoker_df.age 
    estimated_charges = estimate_charges(ages, w, b)
    target = non_smoker_df.charges

    plt.plot(ages, estimated_charges, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    plt.show()
 
# Loss/Cost Function
'''
Calculate Residual = Actual Targer - Our Prediction
Square residual. Since some residuals may contain negative values.
Calculate the average of all squared_residuals. (sum of all sqr_residuals)/N
Result = Root Mean Squared Error (RMSE)
'''

if __name__ == "__main__":

    try_parameter(400, -2000)
        
    
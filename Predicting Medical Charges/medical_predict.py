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
# print(stat_info) 

# Step 3: Visualisation
# Adjusting default style settings
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

fig, (ax1, ax2) = plt.subplots(2, 1)
# Visualise ages in a histogram since age is numeric. Use matplotlib
ax1.hist(medical_df['age'], bins=47, rwidth=0.9) # bins to account for all the ages 18-64. relative width between each bar
ax1.set_title('Distribution of Age')
ax1.set_xlabel('Age')
ax1.set_ylabel('Count')

# Boxplot
ax2.boxplot(medical_df['age'], vert=False)
ax2.set_title('Age Box Plot')
ax2.set_xlabel('Age')

plt.tight_layout()
plt.show()
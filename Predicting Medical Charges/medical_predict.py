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

def plot_graph(*graph_names):
    for graph_name in graph_names:
        if graph_name == "age":
            fig, (ax1, ax2) = plt.subplots(2, 1)

            ax1.hist(medical_df['age'], bins=47, rwidth=0.9) # bins to account for all the ages 18-64. relative width between each bar
            ax1.set_title('Distribution of Age')
            ax1.set_xlabel('Age')
            ax1.set_ylabel('Count')

            ax2.boxplot(medical_df['age'], vert=False)
            ax2.set_title('Age Box Plot')
            ax2.set_xlabel('Age')

            plt.tight_layout()
        
        elif graph_name == "bmi":
            plt.figure()
            plt.hist(medical_df['bmi'], bins=47)
            plt.title('Distribution of BMI')
            plt.xlabel('BMI')
            plt.ylabel('Count')

        elif graph_name == "charges":
            # Split the data
            smoker = medical_df[medical_df['smoker'] == 'yes']['charges']
            non_smoker = medical_df[medical_df['smoker'] == 'no']['charges']
            male = medical_df[medical_df['sex'] == 'male']['charges']
            female = medical_df[medical_df['sex'] == 'female']['charges']

            plt.figure()
            plt.hist(smoker, bins=30, color='green', alpha=0.7, label='Smoker')
            plt.hist(non_smoker, bins=30, color='grey', alpha=0.7, label='Non Smoker')
            plt.title('Annual Medical Charges')
            plt.xlabel('Charges')
            plt.ylabel('Count')
            plt.legend()
        
        elif graph_name == "smoker_by_sex":
            grouped = medical_df.groupby(['smoker', 'sex']).size().unstack()

            grouped.plot(kind='bar', color=['pink', 'blue'], alpha=0.7)
            plt.xlabel('Smoker')
            plt.ylabel('Count')
            plt.title('Smoker Distribution by Sex')
            plt.legend(title='sex')
        elif graph_name == "age_vs_charges":
            fig = px.scatter(medical_df,
                             x='age',
                             y='charges',
                             color='smoker',
                             opacity=0.7,
                             hover_data=['sex'],
                             title='Age vs. Charges')
            fig.update_traces(marker_size=5)

        elif graph_name == "bmi_vs_charges":
            fig = px.scatter(medical_df,
                             x='bmi',
                             y='charges',
                             hover_data=['sex'],
                             opacity=0.7,
                             title='BMI vs Charges')
            fig.update_traces(marker_size=5)

        else:
            print("Graph name not recognised")
    
    plt.show()
    fig.show()

# age, bmi, charges, smoker_by_sex, age_vs_charges, bmi_vs_charges
plot_graph("bmi_vs_charges")

# Count for each category, no or yes in this case
# print(medical_df.smoker.value_counts())
## Problem Statement
The Rain in Australia dataset contains about 10 years of daily weather observations from numerous Australian weather stations. As a data scientist at the Bureau of Meteorology, you are tasked with creating a fully-automated system that can use today's weather data for a given location to predict whether it will rain at the location tomorrow.

1. Download the dataset using Opendatasets
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

## Data Analysis
1. Load the csv into a datafram, *raw_df* using Pandas. 
2. *RainTomorrow* is the target value. As such, drop any rows where *RainTomorrow* is empty
3. Exploratory data analysis can be done during this process, but it will not be performed in this scenario as this is the same dataset used before for Logistic Regression

## Preprocessing Data
1. Create a train/test/validation split
> it's often a good idea to separate the training, validation and test sets with time, so that the model is trained on data from the past and evaluated on data from the "future".
2. Identify input and target columns
> Input: All columns except for first(date) and last(RainTomorrow) columns
> Target: Last, RainTomorrow column
3. Identify numeric and categorical columns
4. Impute (fill) missing numeric values
5. Scale numeric values to the (0, 1) range
6. Encode categorical columns to one-hot-vectors
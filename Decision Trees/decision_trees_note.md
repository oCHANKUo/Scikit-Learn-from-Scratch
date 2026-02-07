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
> numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
> categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

4. Impute (fill) missing numeric values
> from sklearn.impute import SimpleImputer

5. Scale numeric values to the (0, 1) range
> from sklearn.preprocessing import MinMaxScaler
> This step is not necessary for Categorical Data

6. Encode categorical columns to one-hot-vectors
> from sklearn.preprocessing import OneHotEncoder
> Concatanation (*pd.concat*) instead of normal assignment is better for avoiding memory fragmentation
> ex: train_encoded = pd.DataFrame(
            encoder.transform(train_inputs[categorical_cols]),
            columns=encoded_cols,
            index=train_inputs.index
        )
        train_inputs = pd.concat([train_inputs, train_encoded], axis=1)

7. drop the textual categorical columns, so that we're left with just numeric data. Such as Location name

## Model Fitting and Predictions
> from sklearn.tree import DecisionTreeClassifier
> model = DecisionTreeClassifier(random_state=42)
> model.fit(X_train, train_targets)
> train_preds = model.predict(X_train)
> train_probs = model.predict_proba(X_train)
> print(accuracy_score(train_targets, train_preds))
> Accuracy Score = 0.9999797955307714. Training set accuracy is 99%.
> print(model.score(X_val, val_targets)) : Make predictions and calculates accuracy in one step. Acc_score = 0.7914804712436887

> Although the training accuracy is 100%, the accuracy on the validation set is just about 79%.
> It appears that the model has learned the training examples perfect, and doesn't generalize well to previously unseen examples. This phenomenon is called "overfitting".

## Decision Tree Visualisation
> We can visualize the decision tree learned from the training data.
> from sklearn.tree import plot_tree, export_text
> Note the *gini* value in each box. This is the loss function used by the decision tree to decide which column should be used for splitting the data, and at what point the column should be split. A lower Gini index indicates a better split. A perfect split (only one class on each side) has a Gini index of 0.
> Based on the gini index computations, a decision tree assigns an "importance" value to each feature
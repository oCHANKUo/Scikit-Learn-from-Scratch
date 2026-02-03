# Logistic Regression - Notes

## What is Logistic Regression?

- Logistic Regression is a **supervised machine learning algorithm** used to predict **categorical outcomes**, usually binary (0/1, yes/no).
- It predicts the **probability** of a data point belonging to a particular class.

## When to Use

- Target variable is **categorical**.
- Examples:
  - Predict if a patient has a disease (Yes/No)
  - Predict if an email is spam (1) or not spam (0)
  - Predict if a student passes (1) or fails (0)
  
## How It Works

1. Takes input features (can be numeric or one-hot encoded categorical data).
2. Computes a **linear combination** of features: z = w1x1 + w2x2 + ... + b
3. Applies the **sigmoid function** to squash values between 0 and 1: probability = 1 / (1 + e^(-z))
4. Predicts class based on probability:

- If probability > 0.5 → class 1
- If probability ≤ 0.5 → class 0

## Advantages

- Outputs **probabilities**, not just classes.
- Works with **both numeric and categorical features**.
- Simple, easy to implement, interpretable.

## Notes

- Even if inputs are categorical (one-hot encoded), logistic regression is used because the **target is categorical**, unlike linear regression which predicts continuous numbers.
- Can be extended for **multi-class classification** using strategies like One-vs-Rest (OvR) or Softmax.


## Data Analysis Phase

1. Get a quick summary of the dataset using `raw_df.info()` pandas function. It will provide information on data types, column names and not-null counts. We can identify which columns contain Null values.
In this example, the *RainToday* and the target column *RainTomorrow* contain Null values as well.
2. To handle these Null values, we can drop the records where *RainToday* or *RainTomorrow* are Null. It will reduce the row count but will not have Nulls anymore.
Why only these 2 are considered?

> Because they are the most important. Target is *the* most important, but *RainToday* also most likely has a strong influence on the target.

1. Before training a ML model, its good to explore the distributions of the columns and see how they are related to the Target column. This is the aim of the *Data Analysis and Visualization step*.

2. Data Visualisation order (for this specific example):

- Location vs RainToday
- Temperature at 3 pm vs. Rain Tomorrow (can see around which temperatures did they have rain the next day.)
- RainTomorrow vs RainToday (There is a bias for RainTomorrow = No records. Meaning it will be easier predicting RainTomorrow = No than Yes)
- Scatter plot of MinTemp vs MaxTemp
- Temp at 3 pm vs Humidity at 3 pm (RainTomorrow seems to be Yes when humidity is relatively high and temperature is relatively low)

3. Sampling Step: Not necessary but can be helpful for large datasets.


## Machine Learching Phase

1. Split the dataset into 3 parts:

> Training: fit the model
> Validation: tune and evaluate during training
> Testing: final performance check

- First Split: 
  > 80% → training + validation
  > 20% → test set

- Second Split: 
  > 75% → training
  > 25% → validation

- Final Ratio:
  > Training: 60%
  > Validation: 20%
  > Testing: 20%

When considering time series data like this, which involves data over a long period of time, it is a good decision to split data based on date instead of being random. It allows for using data from a specific time period for model fitting while using data from the ""future"" for validation and testing. 
Origin - 2014: Train
2015: Validation
2016 - 2017: Testing

2. Identify Input and Target Columns
> "Date" column can be disregarded since its a model aimed for the future, and the Date column serves no purpose
> MinTemp, MaxTemp matters...similarly understand which columns are useful and which are not
> RainTomorrow column is the Target. Should not be used as an Input.
> Logistic Regression expects you to have a single Target
> InputColumns list and the TargetColumn are created.

> Create seperate data frames for inputs and targets (train, val and test, all)

> Identify Numeric and Categoric columns. After differentiating them, you can use .describe() function to view the statistics.

3. Impute/Fill Missing Numeric Data
> There are several techniques for imputation, but we will use the most basic. Replacing missing values with the average value in the column using SimpleImpute class.

4. Scale Features to a Small Range of Values
> e.g.(0,1) or (-1,1)
> Scaling numeric features ensures that no particular feature has a dispropotionate impact on the models loss.
> Use a MinMaxScaler from sklearn

5. Encoding Categorical Data
> Since machine learning models can only be trained with numeric data, we need to convert catgorical data to numbers. A common technique is *one-hot encoding*

6. Training the Logistic Regression Model
> Saving: It can be useful to save the preprocessed data to the disk, to avoid repeating preprocessing steps. It can be saved in many different formats such as CSV. In this we save it in the parquet format.
> Training data can be saved as is, Target data need to be converted into dataframes (in this example)
> We can read the data back using *pd.read_parquet*
> Logistic Regression is commonly used to solve Binary Classification
> The *solver* tells scikit-learn which algorithm to use to find the best weights
> 'liblinear' is good for small datasets and binary classification problems.

7. Model Evaluation
> To evaluate the Model, caculate the accuracy using *accuracy_score*
> The model achieves an accuracy of 85.1% on the training set. We can visualize the breakdown of correctly and incorrectly classified inputs using a confusion matrix.
> The accuracy of the model on the test and validation set are above 84%, which suggests that our model generalizes well to data it hasn't seen before
> A good way to verify whether a model has actually learned something useful is to compare its results to a "random" or "dumb" model. Two models: one that guesses randomly and another that always return "No". Both of these models completely ignore the inputs given to them.
> Our random model achieves an accuracy of 50% and our "always No" model achieves an accuracy of 77%. Our model is better than a "dumb" or "random" model

8. Making Predictions on a Single Input
> Once the model has been trained to a satisfactory accuracy, it can be used to make predictions on new data.
> We must apply the same transformations applied while training the model:
-- Imputation of missing values using the imputer created earlier
-- Scaling numerical features using the scaler created earlier
-- Encoding categorical features using the encoder created earlier


## Exercise
Initialize the LogisticRegression model with different arguments and try to achieve a higher accuracy. The arguments used for initializing the model are called hyperparameters (to differentiate them from weights and biases - parameters that are learned by the model during training).

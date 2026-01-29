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

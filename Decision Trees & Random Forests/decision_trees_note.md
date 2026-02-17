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

## Hyperparameter Tuning and Overfitting
> The DecisionTreeClassifier accepts several arguments, some of which can be modified to reduce overfitting.
> These arguments are called hyperparameters because they must be configured manually (as opposed to the parameters within the model which are learned from the data.)

1. **max_depth**
> By reducing the maximum depth of the decision tree, we can prevent the tree from memorizing all training examples, which may lead to better generalization
> model = DecisionTreeClassifier(max_depth=3, random_state=42) : while the training accuracy score of the model has gone down, the validation accuracy of the model has increased significantly.
> Max_depth = 7 is the optimal

2. **max_leaf_nodes**
>  Control the size of complexity of a decision tree by limiting the number of leaf nodes. > This allows branches of the tree to have varying depths.
>    model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42).fit(X_train, train_targets)

**optimal max-depth and max-leaf-nodes**
Max Depth            10.000000
Max Leaf Nodes      256.000000
Validation Error      0.154199

## Random Forests

1. Why use Random Forest?
> While tuning the hyperparameters of a single decision tree may lead to some improvements, a much more effective strategy is to combine the results of several decision trees trained with slightly different parameters. This is called a random forest.
> Key idea is that each decision tree in the forest will make different kinds of errors, and upon averaging, many of their errors will cancel out. This idea is also known as the "wisdom of the crowd".

2. How to use in code?
> We'll use the RandomForestClassifier class from sklearn.ensemble
> from sklearn.ensemble import RandomForestClassifier

3. 
> Train Score: 0.9999595910615429
> Validation Score: 0.8560733561604086
> Validation accuracy is much better than before

4. Just like decision tree, random forests also assign an "importance" to each feature, by combining the importance values from individual trees.
> Notice that the distribution is a lot less skewed than that for a single decision tree.

5. Random Forest **Hyperparameters**
> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
> Base model Accuracy:  (0.9999595910615429, 0.8560733561604086)
> **n_estimators**: Controls the number of decision trees in the random forest. Default = 100. Try to have as few as needed.
> n_estimator Accuracy 10:  (0.987331797793672, 0.8431896001392839)
> n_estimator Accuracy 500:  (0.9999797955307714, 0.8571179850269862)

> **max_features**: We can specify the features to be chosen randomly
>  default value auto causes only sqrt(n) out of total features (n) to be chosen randomly at each split.
> max_features_log2:  (0.9999494888269285, 0.8558992513493123)
> max_features_3:  (0.9999393865923142, 0.8511984214497127)
> max_features_6:  (0.9999494888269285, 0.8558992513493123)

> **max_depth** and **max_leaf_nodes**
> Controls the max depth and max nodes for each tree. By default, no max depth.
> max_depth = 5: (0.8197559300117186, 0.8239800359816609)
> max_depth=25: (0.9773810966985897, 0.8560733561604086)
> max_leaf_nodes = 2**5 : (0.8313937042873883, 0.8335558005919563)
> max_leaf_nodes = 2**20 : (0.9999595910615429, 0.8558992513493123)

> **min_samples_split** and **min_samples_leaf**
> By default, the decision tree classifier tries to split every node that has 2 or more
> min_samples_split=3, min_samples_leaf=2: (0.9625813229886451, 0.8571760199640184)
> min_samples_split=100, min_samples_leaf=60: (0.8501333494969087, 0.8454529626835355)

> **min_impurity_decrease**
> Used to control the threshold for splitting nodes
> A node will be split if this split induces a decrease of the impurity (Gini index) greater than or equal to this value. (default = 0)
> min_impurity_decrease=1e-6: (0.9889077463935022, 0.8564215657826011)
> min_impurity_decrease=1e-2: (0.774891906089627, 0.7882885497069235)

> **bootstrap, max_samples**
> By default, a random forest doesn't use the entire dataset for training each decision tree. Instead it applies a technique called bootstrapping.
> Bootstrapping helps the random forest generalize better, because each decision tree only sees a fraction of th training set, and some rows randomly get higher weightage than others.
> bootstap=False: (0.9999797955307714, 0.8570599500899542)

> When bootstrapping is enabled, you can also control the number or fraction of rows to be considered for each bootstrap using max_samples
> max_samples=0.9: 0.9997777508384855, 0.8567117404677616

> **class_weight**
> class_weight='balanced': 0.9999595910615429, 0.8557831814752481
> class_weight={'No': 1, 'Yes': 2}: (0.9999494888269285, 0.8531716093088039)

> **Putting it all together**
> model = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=500, max_features=7, max_depth=30, class_weight={'No': 1, 'Yes': 1.5}): (0.9920596435931628, 0.8564215657826011)
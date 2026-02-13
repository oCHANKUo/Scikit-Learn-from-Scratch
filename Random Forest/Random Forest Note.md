1. Why use Random Forest?
> While tuning the hyperparameters of a single decision tree may lead to some improvements, a much more effective strategy is to combine the results of several decision trees trained with slightly different parameters. This is called a random forest.
> Key idea is that each decision tree in the forest will make different kinds of errors, and upon averaging, many of their errors will cancel out. This idea is also known as the "wisdom of the crowd".

2. How to use in code?
> We'll use the RandomForestClassifier class from sklearn.ensemble
> from sklearn.ensemble import RandomForestClassifier
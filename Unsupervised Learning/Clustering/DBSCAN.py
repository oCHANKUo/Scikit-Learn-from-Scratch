import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

sns.set_style('darkgrid')
iris_df = sns.load_dataset('iris')

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[numeric_cols]

model = DBSCAN(eps=1.1, min_samples=4)
model.fit(X)

if __name__ == '__main__':

    # print(model.labels_)

    sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=model.labels_)
    plt.show()


    print("----------------")
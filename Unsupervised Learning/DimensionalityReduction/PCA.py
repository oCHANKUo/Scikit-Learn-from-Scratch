import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

''' Principal component is a dimensionality reduction technique that uses linear projections of data
 to reduce their dimensions, while attempting to maximize the variance of data in the projection. '''

sns.set_style('darkgrid')
iris_df = sns.load_dataset('iris')

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[numeric_cols]

pca = PCA(n_components=2)
pca.fit(iris_df[numeric_cols])

transformed = pca.transform(iris_df[numeric_cols])


if __name__ == '__main__':
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species']);
    plt.show()


    print("----------------")
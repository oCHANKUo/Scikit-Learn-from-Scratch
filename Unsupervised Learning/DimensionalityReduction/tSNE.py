import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

''' Manifold learning is an approach to non-linear dimensionality reduction. 
Algorithms for this task are based on the idea that the dimensionality of many data sets is only artificially high. '''

sns.set_style('darkgrid')
iris_df = sns.load_dataset('iris')

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[numeric_cols]

tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(iris_df[numeric_cols])

if __name__ == '__main__':
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species'])
    plt.show()


    print("----------------")
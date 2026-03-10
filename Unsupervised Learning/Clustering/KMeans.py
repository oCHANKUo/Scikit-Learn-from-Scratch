import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


sns.set_style('darkgrid')
iris_df = sns.load_dataset('iris')

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = iris_df[numeric_cols]

# model = KMeans(n_clusters=6, random_state=42)
# model.fit(X)


if __name__ == '__main__':


    # sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species')

    # print(model.cluster_centers_)

    # print(model.predict(X))
    # preds = model.predict(X)

    '''sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds)
    centers_x, centers_y = model.cluster_centers_[:,0], model.cluster_centers_[:,2]
    plt.plot(centers_x, centers_y, 'xb')'''    
    # print(model.inertia_)

    options = range(2,11)
    intertias = []
    for n_clusters in options:
        model = KMeans(n_clusters, random_state=42).fit(X)
        intertias.append(model.inertia_)
    
    plt.title("No. of clusters vs Intertia")
    plt.plot(options, intertias, '-o')
    plt.xlabel('No. of clusters (K)')
    plt.ylabel('Intertia')
    plt.show()
    
    print("----------------")
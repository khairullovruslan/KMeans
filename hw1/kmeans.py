import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)



def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = data[np.random.choice(len(data))]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def plot_clusters(data, clusters, centroids, iteration):
    plt.figure(figsize=(6, 6))
    for i in range(len(np.unique(clusters))):
        cluster_points = data[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='центроиды')
    plt.title(f'итерация {iteration}')
    plt.legend()
    plt.show()




def final_plt(data, clusters, centroids):
    num_features = data.shape[1]
    feature_names = ['1', '2', '3', '4']

    plt.figure(figsize=(15, 10))
    subplot_index = 1

    for i in range(num_features):
        for j in range(i + 1, num_features):
            plt.subplot(3, 3, subplot_index)
            subplot_index += 1

            for cluster_id in np.unique(clusters):
                cluster_points = data[clusters == cluster_id]
                plt.scatter(cluster_points[:, i], cluster_points[:, j], label=f'кластер {cluster_id + 1}')

            plt.scatter(centroids[:, i], centroids[:, j], s=200, c='black', marker='X', label='центроиды')

            plt.xlabel(feature_names[i])
            plt.ylabel(feature_names[j])
            plt.legend()

    plt.tight_layout()
    plt.show()


def k_means(data, k, max_iter=10):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for i in range(max_iter):
        clusters = assign_clusters(data, centroids)

        plot_clusters(data, clusters, centroids, i + 1)

        new_centroids = update_centroids(data, clusters, k)

        if np.allclose(centroids, new_centroids):
            print(f"stop - {i}")
            break

        centroids = new_centroids
    final_plt(data, clusters, centroids)



irises = load_iris()
data = irises.data
k = 3
k_means(data, k, max_iter=10)

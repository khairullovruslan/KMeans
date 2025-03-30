from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main():
    irises = load_iris()
    data = irises.data

    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
    plt.grid(True)
    plt.show()


# k = 3

if __name__ == "__main__":
    main()

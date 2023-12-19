from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# Încărcați setul de date MNIST
mnist = fetch_openml('mnist_784')
data = mnist.data / 255.0  # Normalizăm pixelii la intervalul [0, 1]

# Aplicați algoritmul K-means cu k = 10
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(data)

# Obțineți centroizii
centroids = kmeans.cluster_centers_

# Afișați cei 10 centroizi sub formă de imagini
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    centroid_image = centroids[i].reshape(28, 28)  # Imaginile MNIST sunt de dimensiune 28x28
    ax.imshow(centroid_image, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Centroid {i}')

plt.show()

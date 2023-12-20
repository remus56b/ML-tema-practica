import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Încărcați setul de date MNIST
digits = datasets.load_digits()

# Aplicați k-means pe date
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

# Afișați centroizii ca imagini
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()

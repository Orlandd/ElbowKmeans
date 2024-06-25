import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Ganti dengan lokasi file CSV yang benar
file_location = "D:\\Data.csv"


df = pd.read_csv(file_location)


df.columns = df.columns.str.strip()


print("Kolom dalam DataFrame:", df.columns)

# Kolom yang ingin diambil
columns_of_interest = ['Fasilitas Pendidikan', 'Fasilitas Kesehatan', 'Kualitas Udara', 
                       'Sumber Air Minum RT', 'RUANG TERBUKA HIJAU (RTH)', 'Kondisi Lingkungan (Bencana Alam)']


arr = np.array(df[columns_of_interest])


print(arr)


for i, col in enumerate(columns_of_interest):
    plt.figure(figsize=(10, 2))
    y = np.zeros(len(arr))  # Set y constant
    plt.scatter(arr[:, i], y, color='m')
    plt.yticks([])
    plt.xlabel(col)
    plt.title(f'{col} Distribution')
    plt.grid(True)
    plt.show()

k_range = range(1, 11)
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(arr)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('K Value')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method For Optimal K')
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

# Menemukan elbow point
kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
elbow_point = kn.elbow

# Melakukan clustering dengan jumlah cluster yang optimal
km = KMeans(n_clusters=elbow_point, random_state=0, n_init=10)
y_predicted = km.fit_predict(arr)

df['cluster'] = y_predicted

print(df)

centroids = km.cluster_centers_
print("Cluster Centers (Centroids):")
print(centroids)

pca = PCA(n_components=2)
arr_2d = pca.fit_transform(arr)

plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, elbow_point))
for i in range(elbow_point):
    plt.scatter(arr_2d[y_predicted == i, 0], arr_2d[y_predicted == i, 1], color=colors[i], label=f'Cluster {i}')

centers_2d = pca.transform(centroids)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], color='yellow', marker='*', label='Centroid', s=500)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Clustering Hasil dengan PCA')
plt.show()

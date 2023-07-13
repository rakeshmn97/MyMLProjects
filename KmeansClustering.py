import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

Iris_data = pd.read_csv("Dataset/IrisData.csv")
print(Iris_data.head())
Iris_data.Species.unique()
sns.scatterplot(data = Iris_data, x = "SepalLengthCm", y = "PetalWidthCm", hue = Iris_data.Species, palette = "coolwarm_r")
from sklearn.cluster import KMeans

X = Iris_data[["SepalLengthCm","PetalWidthCm"]]
km = KMeans(n_clusters = 3, n_init = 3, init = "random", random_state = 42)
km.fit(X)
y_kmeans = km.predict(X)
print(y_kmeans)
sns.scatterplot(data = Iris_data, x = "SepalLengthCm", y = "PetalWidthCm", hue = y_kmeans, palette = "coolwarm_r")
centers = km.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha = 0.6)
print(km.inertia_)
newdata = [[4.7,0.8]]
y_pred = km.predict(newdata)
print(y_pred)
sns.scatterplot(data = Iris_data, x = "SepalLengthCm", y = "PetalWidthCm")
intertia = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    intertia.append(km.inertia_)

plt.plot(K, intertia, marker="x")
plt.xlabel('k')
plt.xticks(np.arange(15))
plt.ylabel('Intertia')
plt.title('Elbow Method')
plt.show()
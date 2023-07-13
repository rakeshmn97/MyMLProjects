import pandas as pd

# save data under variable name Iris_data
Iris_data = pd.read_csv("Dataset/IrisData.csv")

# display first few rows of data
print(Iris_data.head())
# See species of plants
Iris_data.Species.unique()
X = Iris_data[["SepalLengthCm","PetalLengthCm","PetalWidthCm"]]

# Display shape of data (no. rows, no.columns)
print(X.shape)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

# variables
x = X.iloc[:, 0]
y = X.iloc[:, 1]
z = X.iloc[:, 2]

# axes instance
fig = plt.figure(figsize=(5, 5))
ax = Axes3D(fig)

# color-code species
colors = {'Iris-setosa': 'orange', 'Iris-versicolor': 'grey', 'Iris-virginica': 'lightblue'}

# plot
ax.scatter(x, y, z, s=40, c=Iris_data["Species"].map(colors), cmap=colors, marker='o', alpha=0.5)

# legend
orange_patch = mpatches.Patch(color='orange', label='Iris-setosa')
grey_patch = mpatches.Patch(color='grey', label='Iris-versicolor')
lightblue_patch = mpatches.Patch(color='lightblue', label='Iris-virginica')
ax.legend(handles=[orange_patch, grey_patch, lightblue_patch])

# title
plt.title("Iris plants")

# axes labels
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('PetalLengthCm')
ax.set_zlabel('PetalWidthCm')
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
hc = hc.fit(X)
print(hc.labels_)
from matplotlib.colors import ListedColormap

# create a color map for each cluster
cmap = ListedColormap(["orangered", "lightgreen", "deepskyblue"])

# variables
x = X.iloc[:,0] # SepalLengthCm
y = X.iloc[:,1] # PetalLengthCm
z = X.iloc[:,2] # PetalWidthCm

# axes instance
fig = plt.figure(figsize=(5,5))
ax = Axes3D(fig)

# plot
sc = ax.scatter(x, y, z, s=40, c = hc.labels_, cmap = cmap, marker='o', alpha=1)

# legend
plt.legend(*sc.legend_elements())

#title
plt.title("Hierarchical Clustering")

# axes labels
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('PetalLengthCm')
ax.set_zlabel('PetalWidthCm')
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# define linkage
linked = linkage(X.sample(n=20, random_state=1), 'ward')

# set figure size
plt.figure(figsize=(7, 5))

# dendrogram function
dendrogram(linked,
            orientation='top')

# axis labels
plt.title("Dendrogram")
plt.ylabel("Dissimalirty")
plt.xlabel("Data points")

plt.show()
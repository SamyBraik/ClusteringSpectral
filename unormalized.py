import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


n=600 #Nombre de points
k = 2 #Nombre de clusters
delta = 25
S = np.zeros((n,n)) #Matrice de similarité

#Extraction des données
data = list()
X = list()
Y = list()
g = open("data2.txt","r")
for l1 in g:
    x = (float(l1.rsplit(",")[0]),float(l1.rsplit(",")[1]))
    data.append(x)
    X.append(float(l1.rsplit(",")[0]))
    Y.append(float(l1.rsplit(",")[1]))
g.close()

#Calcul de la matrice de similarité
def GSF(d):
    return np.exp(-delta * d **2)

GSF_v = np.vectorize(GSF)
S = euclidean_distances(data,data)
S = GSF(S)

#Calcul de la matrice laplacienne non normalisée
D = np.diag(np.sum(S, axis=1))
L = np.subtract(D,S)

#Calcul des k premiers vecteurs propres de la matrice laplacienne 
valeurP, vecteurP = np.linalg.eig(L)
indice = np.argsort(valeurP)[:k]
vecteurP = vecteurP[:,indice]

valeurP = np.sort(valeurP)
plt.scatter(np.arange(8),valeurP[:8])
plt.show()
#Application de l'algorithme des k-means
label = KMeans(n_clusters=k).fit(vecteurP).labels_

plt.scatter(X,Y,color="grey")
plt.show()


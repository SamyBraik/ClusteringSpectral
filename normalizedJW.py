import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power

n=600 #Nombre de points
k = 3 #Nombre de clusters
delta = 5.5
S = np.zeros((n,n)) #Matrice de similarité

#Extraction des données
data = list()
X = list()
Y = list()
g = open("data.txt","r")
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

#Calcule de la matrice laplacienne normalisée de J&W
D = np.diag(np.sum(S, axis=1))
L = np.subtract(D,S)
Dm = fractional_matrix_power(D,-0.5)
Ls = np.matmul(np.matmul(Dm,L),Dm)

#Calcul des k premiers vecteurs propres de la matrice laplacienne 
valeurP, vecteurP = np.linalg.eig(L)
vecteurP = vecteurP - np.linalg.norm(vecteurP, axis = 1)[:,None]
indice = np.argsort(valeurP)[:k]
vecteurP = vecteurP[:,indice]

valeurP = np.sort(valeurP)
plt.scatter(np.arange(600),valeurP[:])
plt.show()
#Application de l'algorithme des k-means aux vecteurs propres
label = KMeans(n_clusters=k).fit(vecteurP).labels_

plt.scatter(X,Y,c=label)
plt.show()
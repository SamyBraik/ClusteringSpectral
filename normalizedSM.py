import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles
import pandas as pd
import plotly.express as px

n=600 #Nombre de points
k = 2 #Nombre de clusters
delta = 155
S = np.zeros((n,n)) #Matrice de similarité

#Extraction des données
# data = list()
# X = list()
# Y = list()
# g = open("data.txt","r")
# for l1 in g:
#     x = (float(l1.rsplit(",")[0]),float(l1.rsplit(",")[1]))
#     data.append(x)
#     X.append(float(l1.rsplit(",")[0]))
#     Y.append(float(l1.rsplit(",")[1]))
# g.close()
data, y = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=5)

X,Y=zip(*data)

#Calcul de la matrice de similarité
def GSF(d):
    return np.exp(-delta * d **2)

GSF_v = np.vectorize(GSF)
S = euclidean_distances(data,data)
S = GSF(S)

#Calcule de la matrice laplacienne normalisée de S&M
D = np.diag(np.sum(S, axis=1))
L = np.subtract(D,S)
Lds = np.matmul(np.linalg.inv(D),L)

#Calcul des k premiers vecteurs propres de la matrice laplacienne 
valeurP, vecteurP = np.linalg.eig(L)
indice = np.argsort(valeurP)[:k]
vecteur = vecteurP[:,indice]


# valeurP = np.sort(valeurP)
# plt.scatter(np.arange(7),valeurP[:7])
# plt.show()
#Application de l'algorithme des k-means au vecteur propres
label = KMeans(n_clusters=k).fit(vecteurP).labels_

# plt.scatter(X,Y,c=label)
# plt.show()



# proj_df = pd.DataFrame(vecteurP[: 11].transpose())
# proj_df.head()
# fig = px.imshow(proj_df.corr())
# fig.show()
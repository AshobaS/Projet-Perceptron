import numpy as np
from perceptron_data import bias
from perceptron_data import iris
from numpy.random import rand # Mon python n'arrivait pas à importer pylab
# import matplotlib.pyplot as plt -> Je ne peux pas l'installer "Package python-matplotlib is not available, but is referred to by another package."
# Tout le code concernant maplotlib sera sous commentaire car je ne peux pas le faire fonctionner sur mon terminal

def produitscalaire(x,y): # Fonction calculant le produit scalaire
    taille = len(x)
    somme = 0
    for i in range(taille):
        somme = somme + x[0] * y[0]
    return somme

def genererDonnees(n): # Fonction générant les données
    x1b = (rand(n)*2-1)/2-0.5
    x2b = (rand(n)*2-1)/2+0.5
    x1r = (rand(n)*2-1)/2+0.5
    x2r = (rand(n)*2-1)/2-0.5
    donnees = []
    for i in range(len(x1b)):
        donnees.append(((x1b[i],x2b[i]),False))
        donnees.append(((x1r[i],x2r[i]),True))
    return donnees



def perceptron_apprentissage(N,n): # Fonction appliquant le perceptron aux données d'apprentissage
    w = []
    predicty = 0
    apprentissage = genererDonnees(n)
    nbx = len(apprentissage[0][0])
    for i in range(nbx):
        w.append(0)
    nbligne = len(apprentissage)
    for k in range(1,N):
        for i in range(nbligne):
            x = apprentissage[i][0]
            y = apprentissage[i][1]
            scal = produitscalaire(w,x)
            if scal < 0:
                predicty = False
            else:
                predicty = True
            if predicty != y:
                if y == True:
                    for j in range(nbx):
                        w[j] = w[j] + x[j]
                else:
                    for j in range(nbx):
                        w[j] = w[j] - x[j]
    return w


def perceptron_biais(N,n,biais): # Perceptron avec biais appliqué à bias
    w = []
    predicty = 0
    data = bias
    nbx = 1
    w.append(0)
    nbligne = len(data)
    for i in range(1,N):
        for i in range(nbligne):
            x = data[i][0]
            y = data[i][1]
            scal = w[0] * x + biais
            if scal < 0:
                predicty = False
            else:
                predicty = True
            if predicty != y:
                if y == True:
                    w[0] = w[0] + x
                else:
                    w[0] = w[0] - x
    return w


def perceptron_multiclasse(N,n): # Perceptron multiclasse
    w = []
    predicty = 0
    data = iris
    nbx = len(data[0][0])
    for i in range(nbx):
        w.append(0)
    nbligne = len(data)
    for j in range(1,N):
        for i in range(nbligne):
            x = data[i][0]
            y = data[i][1]
            valx = x[3]
            scal = w[3] * valx
            if scal < w[3]:
                predicty = "iris-setosa"
            else:
                if scal > w[3]:
                    predicty = "iris-virginica"
                else:
                    predicty = "iris-versicolor"
            if predicty != y:
                if y == "iris-setosa":
                    for k in range(nbx):
                        w[k] = w[k] + x[k]
                else:
                    if y == "iris-virginica":
                        for k in range(nbx):
                            w[k] = w[k] + x[k]
                    else:
                        for k in range(nbx):
                            w[k] = w[k] - x[k]
    return w



N = 5
n = 1000
w = []
w = perceptron_apprentissage(N,n)
erreur = 0
test = genererDonnees(n)
nbligne = len(test)
for i in range(nbligne):
    x = test[i][0]
    y = test[i][1]
    scal = produitscalaire(w,x)
    if scal < 0:
        predicty = False
    else:
        predicty = True
    if predicty != y:
        erreur = erreur + 1

#x = np.linspace(0,nbligne,nbligne)
#plt.plot(x, test[x][0], label="Xi test")
#plt.legend()
#plt.plot(x, test[x][1], label="Yi test")
#plt.legend()
#plt.plot(x, w[x], label="Perceptron")
#plt.legend()
#plt.title("Données tests et Perceptron")
#plt.show()

w = perceptron_biais(N * 10,n * 10,1)
#print(w)
w = perceptron_multiclasse(N,n)
#print(w)
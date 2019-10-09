import numpy as np
import matplotlib.pyplot as plt
import time


##Code Modal SNA


#Question 1

#N = np.array([10,100,500,1200,3000,6000,10**4,2*10**4,4*10**4,6*10**4,8*10**4,10**5])
N=int(1e4)
M = int(1000)
rho = 1 #temps de propagation
table=np.zeros(M)
aux=np.zeros((N-1,M))
#res=np.zeros(np.size(N))

table=np.zeros(M) #stockage des résultats
for i in range(1, N):
    T = np.random.exponential((rho *(N - 1))/( i * (N - i)),M)
    table+=T
    aux[i-1,:]+=table

moy=np.mean(table)
std=np.std(table)
erreur=0.96*std/np.sqrt(M)

##plt.scatter(N,res)
##plt.plot(N,2*np.log(N))
##plt.title('Temps de propagation en fonction de n')
##plt.xlabel('Nb de sommets')
##plt.ylabel("Temps d'atteinte moyen")
##plt.show()
print("Question 1 ")
print("Pour rho = ", rho," le temps au bout duquel tout le monde utilise le produit A est:",moy)
print("Intervalle de confiance à 95% :",moy-erreur,moy+erreur)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur)/moy)
print("L'ordre de grandeur théorique: 2*log(N) = ",2*np.log(N)," ")



#Question 2
aux2=np.mean(aux,axis=1)
a=np.linspace(2,N,N-1)
plt.step(aux2,a)
plt.title('Evolution de la propagation du produit A')
plt.xlabel('Temps')
plt.ylabel('Nb de personnes utilisant A')
plt.show()


#Question 3

N = int(1e2)
N_moit=N/2
M = int(1e3)
mu = 1
lam = 1.2
p=mu/(mu+lam)
print('Question 3')

#Méthode de Monte-carlo
print("Méthode naive : Monte-Carlo")

X=np.random.binomial(N,p,size=M)
Y=(X>N_moit)

print("Proba que A_n > B_n sur un graphe en ligne est ",np.mean(Y))
std_emp=np.std(Y)
moy=np.mean(Y)
erreur=0.96*std_emp/np.sqrt(M)
print("Intervalle de confiance à 95% :",moy-erreur,moy+erreur)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur)/moy)


#Méthode de changement de variable
print("Méthode de changement de variable")

print("p=",p)
epsilon = 0.01
q=p+epsilon
print("q=p+epsilon=",q)
r=p*(1-q)/(q*(1-p))

#Simulation de A_n vue comme une loi binomiale
X=np.random.binomial(N,q,size=M)
Y=(X>N_moit)*(r**X)
emp=np.mean(Y)*((1-p)/(1-q))**N
print("Proba que A_n > B_n sur un graphe en ligne est ",emp)
std_emp2=np.std(Y*((1-p)/(1-q))**N)
erreur2=0.96*std_emp2/np.sqrt(M)
print("Intervalle de confiance à 95% :",emp-erreur2,emp+erreur2)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur2)/emp," ")




#Question 4bis: simulation naive et très peu efficace avec 2 boucles 

#Prend longtemps à compiler !!!
print("Question 4")
N = int(1e2)
N_moit = N/2
M = int(1e3)
mu = 1
lam = 3

now = time.time()

res=np.ones((M,2))

for j in range(M):
    for i in range(N-2):
        X=np.random.exponential(1/mu,int(res[j][0]*(N-i-2)))
        Y=np.random.exponential(1/lam,int(res[j][1]*(N-i-2)))
        if min(X)<min(Y):
            res[j][0]+=1
        else:
            res[j][1]+=1

res2=(res[:,0]>res[:,1])
print("Répartition moyenne des produits", np.mean(res,axis=0))

moy=np.mean(res2)
print("Proba que A_n > B_n sur un graphe complet est ",moy)
std_emp=np.std(res2)
erreur=0.96*std_emp/np.sqrt(M)
print("Intervalle de confiance à 95% :",moy-erreur,moy+erreur)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur)/moy)

new_now = time.time()
print("Temps total d'execution, méthode inefficace : ", new_now - now,"\n")


#Question 4:

N = int(1e3)
M = int(1e5)
mu = 1
lam = 4

#Méthode de Monte-Carlo naive
print("Méthode de Monte-Carlo naive")
now = time.time()

res1=np.ones((M,2))
for i in range(N-2):
    a=np.random.random(size=M)
    b=res1[:,0]*mu/(res1[:,0]*mu+res1[:,1]*lam)
    res1[:,0]+=(b>a)
    res1[:,1]+=(b<=a)

res2=(res1[:,0]>res1[:,1])
print("Répartition moyenne des produits: ",np.mean(res1,axis=0))
moy=np.mean(res2)
print("Proba que A_n > B_n sur un graphe complet est ",moy)
std_emp=np.std(res2)
erreur=0.96*std_emp/np.sqrt(M)
print("Intervalle de confiance à 95% :",moy-erreur,moy+erreur)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur)/moy)

new_now = time.time()
print("Temps total d'execution, méthode efficace : ", new_now - now,"\n")

#Méthode de changement de variable"

print("Méthode de changement de variables")
epsilon=np.array([0.5,1,1.5,2,2.5,2.8,3])
mu3=mu+epsilon

for mu2 in mu3:
    res=np.ones((M,2))
    T=np.ones(M)
    for i in range(N-2):
        #Test de Bernoulli de loi b 
        a=np.random.random(size=M)
        b=res[:,0]*mu2/(res[:,0]*mu2+res[:,1]*lam)
        #Renormalisation, voir la formule page 2 du dernier amphi
        T=T*(res[:,0]*mu/(res[:,0]*mu+res[:,1]*lam)*(a<b)+res[:,1]*lam/(res[:,0]*mu+res[:,1]*lam)*(a>=b))
        T=T/(res[:,0]*mu2/(res[:,0]*mu2+res[:,1]*lam)*(a<b)+res[:,1]*lam/(res[:,0]*mu2+res[:,1]*lam)*(a>=b))
        #Transition markovienne 
        res[:,0]+=(b>a)
        res[:,1]+=(b<=a)

    res2=(res[:,0]>res[:,1])
    print(np.mean(res,axis=0))
    moy=np.mean(res2*T)
    print("Proba que A_n > B_n sur un graphe complet est ",moy)
    std_emp=np.std(res2*T)
    erreur=0.96*std_emp/np.sqrt(M)
    print("Intervalle de confiance à 95% :",moy-erreur,moy+erreur)
    print("Taux d'erreur en pourcentage: ", 100*(2*erreur)/moy)

#Méthode d'interaction, sélection, mutation
print("Méthode d'interaction, sélection, mutation")

NbrIter = 5 #Nombre de réalisations indépendantes
Stock = np.zeros(NbrIter)

for nn in range(NbrIter):
    res3=np.ones((M,2))
    #Constante de normalisation (à droite de l'équation (1) TD5)
    estim_constante_normalisation = 1
    #Vecteur de normalisation 
    Gprod=np.ones(M)
    
    for i in range(N-2):
       
        #Application de la fonction de poids
        G=np.exp(1/(1+np.abs(res3[:,0]-res3[:,1])))
        #G=np.exp(-1/2*(np.abs(res3[:,0]-res3[:,1])))
        
        # Update de la constante de normalisation
        estim_constante_normalisation = estim_constante_normalisation * np.mean(G)
        
        # Selection de M ancetres
        indices = np.random.choice(range(M), size=M, p=G/np.sum(G))  
        res3=res3[indices,:]
        
        H=np.exp(-1/(1+np.abs(res3[:,0]-res3[:,1])))
        #H=np.exp(1/2*np.abs(res3[:,0]-res3[:,1])) 
        # Update de la pondération des simulations
        Gprod=Gprod*H

        #Transition markovienne
        a=np.random.random(size=M)
        b=res3[:,0]*mu/(res3[:,0]*mu+res3[:,1]*lam)
        res3[:,0]+=(b>a)
        res3[:,1]+=(b<=a)
        
     

     
    #print(H)
    #print("Constante de normalisation=",estim_constante_normalisation)
    #print("Gprod=",Gprod)
    #print("An: ",np.mean(res3[:,0]),"Bn: ",np.mean(res3[:,1]))
    estim1=(res3[:,0]>res3[:,1])*Gprod
    #print("estim1 =",np.mean(estim1))
    print("Fausse estimation: ",np.mean(res3[:,0]>res3[:,1]))
    moy2=np.mean(estim1*estim_constante_normalisation)
    var2=np.std(estim1*estim_constante_normalisation)
    erreur2=0.96*var2/np.sqrt(M)
    Stock[nn]=moy2
    #print("La probabilité que An>Bn calculée avec la méthode des interactions est",moy2)
    #print("Intervalle de confiance à 95% :",moy2-erreur2,moy2+erreur2)
    #print("Taux d'erreur en pourcentage: ", 100*(2*erreur2)/moy2)

moy3=np.mean(Stock)
var3=np.std(Stock)
erreur3=0.96*var3/np.sqrt(M)
print("moymoy = ",np.mean(Stock))
print("Intervalle de confiance à 95% :",moy3-erreur3,moy3+erreur3)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur3)/moy3)



#Question 5

N = int(1e3)
N_moit = N/2
M = int(1e5)
mu = 1
lam = 4

print("Question 5")
##Graphe en étoile
#On simule le cas où A et B sont sur deux branches
X=np.random.exponential(1/mu,M)
Y=np.random.exponential(1/lam,M)
print("Moyenne empirique de la proba An>Bn est",np.mean(X<Y))
print("Proba théorique est", mu/(mu+lam))

##Graphe en double étoile

#On simule le cas où B est sur une branche et A sur le sommet central de la moitié opposé du graphe.

#N est le nombre de voisins des 2 sommets centraux
res4=np.zeros(M)
#k est le nombre de voisin du sommet contenant A qui n'est pas encore affecté par un produit
k=N
#On simule jusqu'à ce que tous les voisins de ce sommet utilisent un produit.
while k>0:
    a=np.random.random(size=M)
    #La proba que le sommet avec B se propage en premier est lam/(lam+k*mu) (1)
    #La proba que le sommet avec A se propage au sommet central opposé en premier est mu/(lam+k*mu) (2)
    b=k*mu/(lam+k*mu)
    c=mu/(lam+k*mu)
    #Si (1) se produit, res4=1 et on "s'arrete"
    #Si (2) se produit, res4=-1 et on "s'arrete"
    #Sinon, k est incrémenté de -1 et on continue de simuler
    res4=res4+(a>b)*(res4==0)-(a<c)*(res4==0)
    k-=1

##for i in range(np.size(res4)):
##    if res4[i]<0:
##        res4[i]=0
        
res4=(res4+1)/2
res5=1-res4

moy5=np.mean(res5)
print("Probalibité que An>Bn ",moy5)
std_emp5=np.std(res5)
erreur5=0.96*std_emp5/np.sqrt(M)
print("Intervalle de confiance à 95% :",moy5-erreur5,moy5+erreur5)
print("Taux d'erreur en pourcentage: ", 100*(2*erreur5)/moy5)


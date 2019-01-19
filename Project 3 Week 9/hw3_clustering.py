#Importing the libraries
from __future__ import division
from scipy.stats import multivariate_normal as MVN
from scipy.spatial import distance as Dist
import csv as CV
import numpy as np
import sys


np.set_printoptions(precision = 6)

#defining kMeans
def kMeans(data):
    #Defining Cluster numbers
    Cluster_number = 5 
    #maximum number of iterations
    Max_iter = 10
    
    #assigning clusters
    LEN = data.shape[0]
    Zzz = np.zeros(LEN)
    Index_p = np.random.randint(0, LEN, size = Cluster_number)
    
    mu = data[Index_p]
    
    for ITER in range(Max_iter):
        for i, Dd in enumerate(data):
            Placeholder1 = np.linalg.norm(mu - Dd, 2, 1)
            Zzz[i] = np.argmin(Placeholder1)
        
        n = np.bincount(Zzz.astype(np.int64), None, Cluster_number)      
        
        for k in range(Cluster_number):
            Index_p = np.where(Zzz == k)[0]
            mu[k] = (np.sum(data[Index_p], 0))/float(n[k])

        #Output
        outPath = "centroids-{:g}.csv".format(ITER + 1)
        with open(outPath, "w") as file:
            writer = CV.writer(file, delimiter=',', lineterminator = '\n')
            for ValuE in mu:
                writer.writerow(ValuE)

#defining emGMM
def emGMM(data):
#Fixing class numbrs
    classes = 5 
    Max_iter = 10
    LEN = data.shape[0]
    Dmention = data.shape[1]
    Matrix_sig = np.eye(Dmention)
#ID matrix
    sigma = np.repeat(Matrix_sig[:,:,np.newaxis],classes,axis=2)   
#uniform distribution
    class_pi = np.ones(classes)*(1/classes) 
    Gamma = np.zeros((LEN, classes))
    norm_Gamma = np.zeros((LEN, classes))
    Index_p = np.random.randint(0, LEN, size = classes)
    mu = data[Index_p]


    for ITER in range(Max_iter):
        #Expec of EM Alg
        for k in range(classes):
            Matrix_sig_INV = np.linalg.inv(sigma[:,:,k])
            Matrix_sig_INV_2 = (np.linalg.det(sigma[:,:,k]))**-0.5
            for Indexes in range(LEN):
                Dd = data[Indexes,:]
                Placeholder1 = (((Dd-mu[k]).T).dot(Matrix_sig_INV)).dot(Dd-mu[k])
                Gamma[Indexes, k] = class_pi[k]*((2*np.pi)**(-Dmention/2))*Matrix_sig_INV_2*np.exp(-0.5*Placeholder1)
            for Indexes in range(LEN):
                Total1 = Gamma[Indexes,:].sum()
                norm_Gamma[Indexes,:] = Gamma[Indexes,:]/float(Total1)
        
#Maximisation EM ALG
        Max_x = np.sum(norm_Gamma, axis=0)
        class_pi = Max_x/float(LEN)
        for k in range(classes):
            mu[k] = ((norm_Gamma[:,k].T).dot(data))/Max_x[k]
        for k in range(classes):
            Placeholder1 = np.zeros((Dmention, 1))
            Placeholder2 = np.zeros((Dmention, Dmention))
            for Indexes in range(LEN):
                Dd = data[Indexes,:]
                Placeholder1[:,0] = Dd - mu[k]                
                Placeholder2 = Placeholder2 + norm_Gamma[Indexes, k]*np.outer(Placeholder1, Placeholder1)
            sigma[:,:,k] = Placeholder2/float(Max_x[k]) 

#Output
        outPath = "pi-{:g}.csv".format(ITER + 1)
        with open(outPath, "w") as file:
            writer = CV.writer(file, delimiter=',', lineterminator='\n')
            for ValuE in class_pi:
                writer.writerow([ValuE])
        #Write output to file
        outPath = "mu-{:g}.csv".format(ITER + 1)
        with open(outPath, "w") as file:
            writer = CV.writer(file, delimiter=',', lineterminator='\n')
            for ValuE in mu:
                writer.writerow(ValuE)
        #Write output to file
        for k in range(classes):
            outPath = "Sigma-{:g}-{:g}.csv".format(k + 1, ITER + 1)
            with open(outPath, "w") as file:
                writer = CV.writer(file, delimiter=',', lineterminator = '\n')
                for ValuE in sigma[:,:,k]:
                    writer.writerow(ValuE)


#defining main                
def main():  
    filename = np.genfromtxt(sys.argv[1], delimiter=',')
    
    kMeans(filename)
    emGMM(filename)
    
if __name__ == "__main__":
    main()
    print(sys.argv)
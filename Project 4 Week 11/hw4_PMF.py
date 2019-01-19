#Importing the libraries
from __future__ import division
import numpy as np
import sys
import csv as CV
import os

#defining numpy
np.set_printoptions(precision = 7)
np.random.seed(50)

#defining PMF probabilistic matrix factorization
def PMF(train_data):
    lam = 2
    sigma2 = 0.1
    d = 5 
    Phi = 0
    Max_itr = 50
    itr_Print = [10, 25, 50]
    OBJ_f = np.zeros((Max_itr, 1))
    
#defining the two matrices
    ZzZ1 = int(np.amax(train_data[:,0]))
    ZzZ2 = int(np.amax(train_data[:,1]))
    
#U and V matrices    
    Vmatrices = np.random.normal(Phi, np.sqrt(1/lam), (ZzZ2, d))
    Umatrices = np.zeros((ZzZ1, d))
    
#creating umatrices    
    U_matr_id = []
    for Z in range(ZzZ1):
        Placeholder1 = train_data[train_data[:,0] == Z + 1][:,1] 
        Placeholder2 = Placeholder1.astype(np.int64)
        #append
        U_matr_id.append(Placeholder2)        

#creating vmatrices    
    V_matr_id = []
    for Y in range(ZzZ2):
        Placeholder1 = train_data[train_data[:,1] == Y + 1][:,0] 
        Placeholder2 = Placeholder1.astype(int)
        #append
        V_matr_id.append(Placeholder2)
     
#Form ZzZ_matrices   
    ZzZ_matrices = np.zeros((ZzZ1, ZzZ2))
    for train_Value in train_data:
        Rows = int(train_Value[0])
        Columns = int(train_Value[1])
        ZzZ_matrices[Rows - 1, Columns - 1] = train_Value[2]

    for Iters in range(Max_itr):
#Umatrices 
        for Z in range(ZzZ1):
            Placeholder1 = lam * sigma2 * np.eye(d)
            Placeholder2 = Vmatrices[U_matr_id[Z] - 1]
            Placeholder3 = (Placeholder2.T).dot(Placeholder2)
            Placeholder4 = np.linalg.inv(Placeholder1 + Placeholder3)
            Placeholder5 = ZzZ_matrices[Z, U_matr_id[Z] - 1]
            Placeholder6 = (Placeholder2 * Placeholder5[:,None]).sum(axis = 0)
            Placeholder7 = Placeholder4.dot(Placeholder6)
            Umatrices[Z] = Placeholder7
            
#Vmatrices
        for Y in range(ZzZ2):
            Placeholder1 = lam * sigma2 * np.eye(d)
            Placeholder2 = Umatrices[V_matr_id[Y] - 1]
            Placeholder3 = (Placeholder2.T).dot(Placeholder2)
            Placeholder4 = np.linalg.inv(Placeholder1 + Placeholder3)
            Placeholder5 = ZzZ_matrices[V_matr_id[Y] - 1, Y]
            Placeholder6 = (Placeholder2 * Placeholder5[:, None]).sum(axis = 0)
            Placeholder8 = Placeholder4.dot(Placeholder6)
            Vmatrices[Y] = Placeholder8    

#map
        Prod_3 = 0
        Prod_1 = lam * 0.5 * (((np.linalg.norm(Umatrices, axis = 1))**2).sum())     
        Prod_2 = lam * 0.5 * (((np.linalg.norm(Vmatrices, axis = 1))**2).sum())
        
        for train_Value in train_data:
            Z = int(train_Value[0])
            Y = int(train_Value[1])
            Prod_3 = Prod_3 + (train_Value[2] - np.dot(Umatrices[Z - 1,:], Vmatrices[Y - 1,:]))**2
        Prod_3 = Prod_3/(2 * sigma2)
        OBJ_f[Iters] = - Prod_3 - Prod_1 - Prod_2             
            
        if Iters+1 in itr_Print:
#Output
            Out_p = "U-{:g}.csv".format(Iters + 1)
            with open(Out_p, "w") as Filename:
                writer = CV.writer(Filename, delimiter=',', lineterminator='\n')
                for train_Value in Umatrices:
                    writer.writerow(train_Value)   
#output_
            Out_p = "V-{:g}.csv".format(Iters + 1)
            with open(Out_p, "w") as Filename:
                writer = CV.writer(Filename, delimiter = ',', lineterminator = '\n')
                for train_Value in Vmatrices:
                    writer.writerow(train_Value)             

#output_
    Out_p = "objective.csv"
    with open(Out_p, "w") as Filename:
        writer = CV.writer(Filename, delimiter = ',', lineterminator = '\n')
        for train_Value in OBJ_f:
            writer.writerow(train_Value) 
            

  #defining main                  
def main():  
    final_File = np.genfromtxt(sys.argv[1], delimiter = ',')
     #probabilistic matrix factorization  
    PMF(final_File)
    
    #final
if __name__ == "__main__":
    main()
    #end of proj.
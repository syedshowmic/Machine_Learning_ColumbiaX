from __future__ import division
import numpy as np
import sys
from scipy.stats import multivariate_normal as mvn

X_train = np.genfromtxt(sys.argv[1], delimiter = ",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter = ",")

def pluginClassifier(X_train, y_train, X_test): 
    n = len(set(y_train))
    
    Prob_prior = [y_train.tolist().count(val)/n for val in set(y_train)]
    mean_class = []
    cov_class = []
    conditional_class = []
    
    for val in set(y_train):
        mean_class.append(np.mean(X_train[np.where(y_train==val)[0]], axis = 0))
        cov_class.append(np.cov(X_train[np.where(y_train==val)[0]], rowvar = 0))
        conditional_class.append(mvn(mean=mean_class[-1], cov = cov_class[-1]))
    Prob_posterior = []
    
    for test in X_test:
        Numrator = [Prob_prior[i]*conditional_class[i].pdf(test) for i in range(n)]
        prob = [Numrator[i]/sum(Numrator) for i in range(n)]
        Prob_posterior.append(prob)
    
    return np.array(Prob_posterior)
 

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter = ",") # write output to file
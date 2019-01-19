#HW 1

#Importing the libraries
import sys
import numpy as np

#Creating input code for lambda
lambda_input = int(sys.argv[1])

#Creating input code for Sigma
sigma2_input = float(sys.argv[2])

#Building the training set and test set
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])


X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
X_train = np.matrix(X_train)
y_train = np.transpose(np.matrix(y_train))
X_test = np.matrix(X_test)

## Solution for Part 1
def part1(y_train, X_train):
    z = np.shape(X_train)[1]
    term_1 = np.linalg.inv(((lambda_input) * (np.identity(z)))
                + (np.transpose(X_train) * X_train))
    term_2 = np.transpose(X_train) * y_train
    return (term_1 * term_2)


wRR = part1(y_train, X_train)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

## Solution for Part 2
def part2(y_train, X_train, X_test):
    z = np.shape(X_train)[1]
    indice = list(range(1, len(X_test) + 1))
    sigma = np.linalg.inv(lambda_input*np.identity(z) + 
                          sigma2_input**(-1) * np.transpose(X_train) * X_train)
    loc = []
    while len(loc) <= 10:
        VAR = []
        for i in range(len(X_test)):
            sigma_temp = np.linalg.inv(np.linalg.inv(sigma) + 
                                       sigma2_input**(-1) * np.transpose(X_test[i]) * X_test[i])
            VAR.append(sigma2_input + X_test[i] * sigma_temp * np.transpose(X_test[i]))
        Index = np.argmax(VAR)
        sigma = np.linalg.inv(np.linalg.inv(sigma) + 
                                       sigma2_input**(-1) * np.transpose(X_test[Index]) * X_test[Index])
        X_test = np.delete(X_test, Index, 0)
        loc.append(indice[Index])
        indice.remove(indice[Index])
    
    return loc

active = part2(y_train, X_train, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
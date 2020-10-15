import csv
from numpy import linalg as LA
import numpy as np
import os


def file_read(csv_file_name):
    # file_read reads file and enters it into an array
    dir_name = os.path.dirname(__file__)
    file_name = os.path.join(dir_name, csv_file_name)
    reader = csv.reader(open(file_name), delimiter=",")
    x_matrix_raw = array(list(reader)).astype("str")
    return x_matrix_raw


def p_extraction_lagrange_multiplier(x_matrix, x_matrix_transpose):
    # p_extraction_lagrange_multiplier returns p_vector,
    # the solution to the optimization of p(T)X(T)Xp such that norm(p) = 1
    x_transpose_x = x_matrix_transpose @ x_matrix  # Matrix multiplication to determine XTransposeX matrix
    eig_values, eig_vectors = LA.eig(x_transpose_x)  # Eigvenvalues and Eigenvectors of XTX
    max_eig_value = eig_values.max()
    eig_values = list(eig_values.real)
    max_eigen_value_index = eig_values.index(max_eig_value)
    p_vector = np.array((eig_vectors[:, max_eigen_value_index]).real)  # eigenvector of XTX with largest eigenvalue
    return p_vector, x_matrix, x_matrix_transpose


csvFileName = 'TEPdataProc1dataCSV.csv'
XMatrixRaw = file_read(csvFileName)
XMatrix = np.array(delete(XMatrixRaw, 0, 0))                    # removes headers of the data
XMatrix = np.array(delete(XMatrix, XMatrix.shape[1] - 1, 1))    # removes last column with sample time data
rows, cols = XMatrix.shape[0], XMatrix.shape[1]
XMatrix = XMatrix.astype(np.float)
XMatrixTranspose = np.array(XMatrix.transpose())
tol = 1e-12                                                      # set tolerance for relative error to make sure norm(p) == 1
PCArray = []
pVector, xMatrix, xMatrixTranspose = p_extraction_lagrange_multiplier(XMatrix, XMatrixTranspose)  # loading vector p, eigenvector for PC1
tVector = xMatrix @ pVector                                                                       # Matrix multiplication to determine latent score vector t
normPVector = linalg.norm(pVector)                               # ensures that the eigenvector has the highest magnitude of 1
relativeError = abs(1 - normPVector)
count = 0

while len(PCArray) < cols + 2:
    new_relative_error = abs(1 - normPVector)       #determines convergence criteria if the p vector gives the largest veriance
    pVector = pVector.reshape(cols, 1)
    pVectorTranspose = pVector.transpose()
    tVector = np.resize(tVector, (rows, 1))
    xMatrix = xMatrix - np.asarray(tVector @ pVectorTranspose)      #Deflate xMatrix to extract next loading vector
    xMatrixTranspose = xMatrix.transpose()
    if count < 2:
        PCArray.append(pVector.transpose())
        count += 1
        continue
    else:
        if new_relative_error < tol:
            PCArray.append(pVector.transpose())
            pVector, xMatrix, xMatrixTranspose = p_extraction_lagrange_multiplier(xMatrix, XMatrixTranspose)
        else:
            print("Something went wrong, please check your data set again.")
PCArray.pop(0)
PCArray.pop(0)
with open('PrincipleComponents.csv', 'w') as f:
    for item in PCArray:
        f.write(str(item) + ", \n\n")

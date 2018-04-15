import cv2
import os
import numpy as np
import load_data
import test 
import pickle
import random
def normalization(x, mean):

    x = x-mean
    return x

def main():
    
    trainx, trainy, testx, testy = load_data.load_data()
    # print(trainx.shape, trainy.shape)
    # print(testx.shape, testy.shape)
    # print(trainx.shape, "\n", trainx, "\n", trainy.shape, "\n", trainy, "\n", testx.shape, "\n", testx, "\n", testy.shape, "\n", testy)
    clumped_test = np.concatenate([np.asarray(testx).T, np.asarray(testy)], axis=1)
    # print(clumped_test)
    clumped_train = np.concatenate([np.asarray(trainx).T, np.asarray(trainy)], axis=1)
    clumped_test = np.take(clumped_test,np.random.permutation(clumped_test.shape[0]),axis=0,out=clumped_test)
    trainn = np.concatenate([clumped_train, clumped_test[:int(.75*len(clumped_test)),:]], axis=0)
    testt = clumped_test[int(.75*len(clumped_test)):,:]

    trainx = trainn[:,:-1].T
    trainy = trainn[:,-1:]
    testx = testt[:,:-1].T
    testy = testt[:,-1:]
    # print(trainx.shape, trainy.shape)
    # print(testx.shape, testy.shape)

    rows, cols = trainx.shape
    mean = trainx.mean(axis = 1)
    mean = mean.reshape(rows, 1)
    new_trainx = normalization(trainx, mean)
    k = 32
    cov_matrix = np.cov(new_trainx.T)
    eigenval, eigenvec = np.linalg.eig(cov_matrix)
    ind = eigenval.argsort()[::-1]   
    eigenval = eigenval[ind]
    eigenvec = eigenvec[:,ind]
    print(eigenvec.shape)
    sigma = eigenvec[0:k, :]
    sigma = sigma.T
    eigen_faces = np.dot(sigma.T , new_trainx.T)
    signature_faces = eigen_faces.dot(new_trainx)
    test_signature_faces = eigen_faces.dot(testx)

    # Pickle signature faces
    pickle_out = open("train_data.pickle", "wb")
    pickle.dump(signature_faces.T, pickle_out)
    pickle_out.close()

    pickle_out = open("test_data.pickle", "wb")
    pickle.dump(test_signature_faces.T, pickle_out)
    pickle_out.close()

    print("shape = ", signature_faces.shape)
    test.test(trainy, mean, testx, testy, eigen_faces, signature_faces)


if __name__ == '__main__':
    main()
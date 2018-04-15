import cv2
import os
import numpy as np

imposter = False

def accuracy(prediction, testy):

    cnt = 0
    for x, y in zip(prediction, testy):
        print(x, "\t", y)
        if x == y or x == -1:
            cnt += 1
    print("Accuracy = ", (cnt/testy.shape[0])*100)

def test(trainy, mean, testx, testy, eigen_face, signature_face):
    prediction = []
    print(signature_face.T.shape)
    for test_image in testx.T:

        test_image = test_image.reshape(testx.shape[0],1) - mean
        final_eigenface = eigen_face.dot(test_image)
        min_dist = np.linalg.norm(signature_face.T[0].reshape(final_eigenface.shape) - final_eigenface)
        index = 0
        count = 0

        for col in signature_face.T:
            v = col.reshape(final_eigenface.shape)
            dist = np.linalg.norm(v - final_eigenface)
            if dist < min_dist:
                min_dist = dist
                index = count
            count += 1

        prediction.append(trainy[index][0])

    accuracy(prediction, testy)

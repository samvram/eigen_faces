# part a

import matplotlib.cm as cm
import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot as plt1
from scipy import misc
from sklearn.linear_model import LogisticRegression


def get_feature_matrix(path, num, average_face, eigen_face):
    testing_labels, testing_data = [],[]

    with open(path) as f:
        for line in f:
            im = misc.imread(line.strip().split()[0])
            testing_data.append(im.reshape(2500,))
            testing_labels.append(line.strip().split()[1])

    # part b
    testing_data, testing_labels = np.array(testing_data, dtype=float), np.array(testing_labels, dtype=int)

    # part d
    mean_subtracted_data = testing_data
    for x in range(0, testing_data.shape[0]):
        mean_subtracted_data[x, :] = mean_subtracted_data[x, :] - average_face

    # Part G
    r=num
    F_feature_matrix_test = mean_subtracted_data.dot(np.transpose(eigen_face[:r,:]))
    return(F_feature_matrix_test, testing_labels)


train_labels, train_data = [],[]

with open('faces/train.txt') as f:
    for line in f:
        im = misc.imread(line.strip().split()[0])
        train_data.append(im.reshape(2500,))
        train_labels.append(line.strip().split()[1])

# part b
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
plt.imshow(train_data[10,:].reshape(50,50), cmap = cm.Greys_r)
plt.title('The 10th Data')
# plt.show()

# part c
average_face = np.zeros((2500,), dtype = float)
for x in range(0, train_data.shape[0]):
    average_face = average_face + train_data[x, :]

average_face = average_face/train_data.shape[0]

plt.figure()
plt.title('Average Face')
plt.imshow(average_face.reshape(50, 50), cmap = cm.Greys_r)
# plt.show()

# part d
mean_subtracted_data = train_data
for x in range(0, train_data.shape[0]):
    mean_subtracted_data[x, :] = mean_subtracted_data[x, :] - average_face

plt.figure()
plt.title('Mean Subtracted Face')
plt.imshow(mean_subtracted_data[10,:].reshape(50, 50), cmap = cm.Greys_r)

# part e
U, Sigma, Vt = np.linalg.svd(mean_subtracted_data)
plt.figure()
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    plt.imshow(Vt[i, :].reshape(50, 50), cmap = cm.Greys_r)
plt.title('10 Eigen Faces')
plt.show()

# part f
# low_ranx_approximation = np.zeros(mean_subtracted_data.shape)
# error = np.zeros(200)
# for r in range(1, 200):
#     # print('Now evaluating r = '+str(r))
#     Zeta = np.zeros((r,r))
#     for i in range(0, r):
#         Zeta[i, i] = Sigma[i]
#     low_ranx_approximation = (U[:,:r].dot(Zeta)).dot(Vt[:r,:])
#     for i in range(0, mean_subtracted_data.shape[0]):
#         for j in range(0, mean_subtracted_data.shape[1]):
#             aij = (low_ranx_approximation[i][j] - mean_subtracted_data[i][j])
#             error[r-1] = error[r-1] + aij*aij
#     error[r-1] = np.sqrt(error[r-1])
# plt1.figure()
# plt1.plot(np.linspace(1,200,200), error)
# plt1.title('Error vs r')
# plt1.xlabel('r ->')
# plt1.ylabel('||X-X_r|| ->')
# plt1.show()

# Part G
r=10
print('r = '+str(r))
F_feature_matrix_train = mean_subtracted_data.dot(np.transpose(Vt[:r,:]))
F_feature_matrix_test, test_labels = get_feature_matrix('faces/test.txt', r, average_face, Vt)

# Part H
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(F_feature_matrix_train, train_labels)

test_acc = logistic_regression_model.score(F_feature_matrix_test, test_labels)
print('Testing Accuracy : '+str(test_acc*100)+' %')

# Part I
Accuracy = np.zeros(200)
for r in range(1,200):
    F_feature_matrix_test, test_labels = get_feature_matrix('faces/test.txt', r, average_face, Vt)
    F_feature_matrix_train = mean_subtracted_data.dot(np.transpose(Vt[:r, :]))
    logistic_regression_model.fit(F_feature_matrix_train, train_labels)
    Accuracy[r-1] = logistic_regression_model.score(F_feature_matrix_test, test_labels)*100

plt1.figure()
plt1.title('Accuracy vs r')
plt1.xlabel('r ->')
plt1.ylabel('Accuracy(%) ->')
plt1.plot(np.linspace(1,200,200),Accuracy)
plt1.show()
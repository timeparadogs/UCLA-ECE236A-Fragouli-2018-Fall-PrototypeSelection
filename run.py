# file that generates all the figures

import classifier.py

X_GMM = (TrainData_GMM)[0]
y_GMM = (TrainData_GMM)[1]
lambda_GMM = 1 / (X_GMM.shape)[0]

# explicitly calculate the whole n x n distance matrix
dist_mat_GMM = squareform(pdist(X_GMM, metric="euclidean"))
dist_mat_GMM = dist_mat_GMM.flatten()
dist_mat_GMM = [distance for distance in dist_mat_GMM if distance > 0]
percentile2_GMM = np.percentile(dist_mat_GMM, q=2)
percentile40_GMM = np.percentile(dist_mat_GMM, q=40)
percentile60_GMM = np.percentile(dist_mat_GMM, q=60)
epsilon_range_GMM = np.linspace(start=percentile2_GMM, stop=percentile60_GMM, num=100)

test_error_holder_GMM = []
cover_error_holder_GMM = []
prot_number_holder_GMM = []
for epsilon in epsilon_range_GMM:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_GMM, y_GMM, epsilon, lambda_GMM, k=4, verbose=False)
    test_error_holder_GMM.append(test_error)
    cover_error_holder_GMM.append(cover_error)
    prot_number_holder_GMM.append(prots)

plt.figure()
plt.scatter(prot_number_holder_GMM, test_error_holder_GMM, c='r', marker='o', alpha=0.66)
plt.title('Test Error on my Gaussian Mixtures')
plt.ylabel('Average Test Error')
plt.xlabel('Average Number of Prototypes')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.savefig('gmmtest.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_GMM, cover_error_holder_GMM, c='b', marker='o', alpha=0.66)
plt.title('Cover Error on my Gaussian Mixtures')
plt.ylabel('Average Cover Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.savefig('gmmcover.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_GMM, test_error_holder_GMM, c='r', marker='o', alpha=0.66, label='Test Error')
plt.scatter(prot_number_holder_GMM, cover_error_holder_GMM, c='b', marker='o', alpha=0.66, label='Cover Error')
plt.title('Error on my Gaussian Mixtures Dataset')
plt.ylabel('Average Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.legend()
plt.savefig('gmmboth.png')
plt.show()

#################################################################################################################

TrainData_IRIS = load_iris(return_X_y=True)
X_IRIS = (TrainData_IRIS)[0]
y_IRIS = (TrainData_IRIS)[1]
lambda_IRIS = 1 / (X_IRIS.shape)[0]

# explicitly calculate the whole n x n distance matrix
dist_mat_IRIS = squareform(pdist(X_IRIS, metric="euclidean"))
dist_mat_IRIS = dist_mat_IRIS.flatten()
dist_mat_IRIS = [distance for distance in dist_mat_IRIS if distance > 0]
percentile2_IRIS = np.percentile(dist_mat_IRIS, q=2)
percentile40_IRIS = np.percentile(dist_mat_IRIS, q=40)
percentile60_IRIS = np.percentile(dist_mat_IRIS, q=60)
epsilon_range_IRIS = np.linspace(start=percentile2_IRIS, stop=percentile60_IRIS, num=100)

test_error_holder_IRIS = []
cover_error_holder_IRIS = []
prot_number_holder_IRIS = []
for epsilon in epsilon_range_IRIS:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_IRIS, y_IRIS, epsilon, lambda_IRIS, k=4, verbose=False)
    test_error_holder_IRIS.append(test_error)
    cover_error_holder_IRIS.append(cover_error)
    prot_number_holder_IRIS.append(prots)

plt.figure()
plt.scatter(prot_number_holder_IRIS, test_error_holder_IRIS, c='r', marker='o', alpha=0.66)
plt.title('Test Error on the Iris Dataset')
plt.ylabel('Test Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.savefig('iristest.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_IRIS, cover_error_holder_IRIS, c='b', marker='o', alpha=0.66)
plt.title('Cover Error on the Iris Dataset')
plt.ylabel('Cover Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.savefig('iriscover.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_IRIS, test_error_holder_IRIS, c='r', marker='o', alpha=0.66, label='Test Error')
plt.scatter(prot_number_holder_IRIS, cover_error_holder_IRIS, c='b', marker='o', alpha=0.66, label='Cover Error')
plt.title('Error on the Iris Dataset')
plt.ylabel('Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.legend()
plt.savefig('irisboth.png')
plt.show()

TrainData_CANC = load_breast_cancer(return_X_y=True)
X_CANC = (TrainData_CANC)[0]
y_CANC = (TrainData_CANC)[1]
lambda_CANC = 1 / (X_CANC.shape)[0]

# explicitly calculate the whole n x n distance matrix
dist_mat_CANC = squareform(pdist(X_CANC, metric="euclidean"))
dist_mat_CANC = dist_mat_CANC.flatten()
dist_mat_CANC = [distance for distance in dist_mat_CANC if distance > 0]
percentile2_CANC = np.percentile(dist_mat_CANC, q=2)
percentile40_CANC = np.percentile(dist_mat_CANC, q=40)
percentile50_CANC = np.percentile(dist_mat_CANC, q=50)
epsilon_range_CANC = np.linspace(start=percentile2_CANC, stop=percentile40_CANC, num=100)

test_error_holder_CANC = []
cover_error_holder_CANC = []
prot_number_holder_CANC = []
for epsilon in epsilon_range_CANC:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_CANC, y_CANC, epsilon, lambda_CANC, k=4, verbose=False)
    test_error_holder_CANC.append(test_error)
    cover_error_holder_CANC.append(cover_error)
    prot_number_holder_CANC.append(prots)

plt.figure()
plt.scatter(prot_number_holder_CANC, test_error_holder_CANC, c='r', marker='o', alpha=0.66)
plt.title('Test Error on the Breast Cancer Dataset')
plt.ylabel('Test Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.savefig('canctest.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_CANC, cover_error_holder_CANC, c='b', marker='o', alpha=0.66)
plt.title('Cover Error on the Breast Cancer Dataset')
plt.ylabel('Cover Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.savefig('canccover.png')
plt.show()

plt.figure()
plt.scatter(prot_number_holder_CANC, test_error_holder_CANC, c='r', marker='o', alpha=0.66, label='Test Error')
plt.scatter(prot_number_holder_CANC, cover_error_holder_CANC, c='b', marker='o', alpha=0.66, label='Cover Error')
plt.title('Error on the Breast Cancer Dataset')
plt.ylabel('Average Error')
plt.ylim(-0.1,1.1)
plt.xlim(0)
plt.grid(linestyle='--')
plt.xlabel('Average Number of Prototypes')
plt.legend()
plt.savefig('cancboth.png')
plt.show()
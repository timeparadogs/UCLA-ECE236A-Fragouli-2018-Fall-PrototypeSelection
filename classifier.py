'''Libraries for Prototype selection'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics

class classifier():
    '''Contains functions for prototype selection'''
    def __init__(self, X, y, epsilon_, lambda_ ):
        '''Store data points as unique indexes, and initialize 
        the required member variables eg. epsilon, lambda, 
        interpoint distances, points in neighborhood'''

        self.X = X
        self.y = y
        assert( (self.X.shape)[0] == (self.y.shape)[0] )
        self.size = (self.X.shape)[0]
        self.epsilon_ = epsilon_
        self.lambda_ = 1 / self.size
        

    '''Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union, etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose.'''


    def getLabels(self):
        '''Get the labels in the dataset'''
        self.L = set(self.y)
        return(set(self.y)) 

    def getNumOfLabels(self):
        '''Get the number of classes or labels'''
        self.sizeL = len(self.L)
        return(self.sizeL)    

    def instantiate_dataDict_0(self, desiredLabel):
        '''Get a subset of the X data according to the label'''
        indices = [index for index, labels in enumerate(self.y) if labels == desiredLabel]
        X_l = self.X[indices]
        y_l = self.y[indices]
        size = (X_l.shape)[0]
        indices_NOT_l = [index for index, labels in enumerate(self.y) if labels != desiredLabel]
        X_NOT_l = self.X[indices_NOT_l]
        y_NOT_l = self.y[indices_NOT_l]
        size_NOT_l = (X_NOT_l.shape)[0]
        data = {
            "X_l": X_l,
            "y_l": y_l,
            "indices_l": indices,
            "size_l": size,
            "X_NOT_l": X_NOT_l,
            "y_NOT_l": y_NOT_l,
            "indices_NOT_l": indices_NOT_l,
            "size_NOT_l": size_NOT_l,
        }
        return(data)

    def checkPointInNeighborhood(self, x0, x_test, neighborhoodType="l2_ball"):
        result = float('nan')
        if neighborhoodType == "l2_ball":
            result = (np.linalg.norm((x0-x_test), ord=2) <= self.epsilon_)

        return(result)

    def C_l_j_constructor_1(self, label, neighborhoodType="l2_ball"):
        data = self.instantiate_dataDict_0(desiredLabel=label)
        X_l = data["X_l"]
        X_NOT_l = data["X_NOT_l"]
        C_l_j_sets = []
        for x_j_l in X_l:
            temp_set_holder = 0
            for x_n_NOT_l in X_NOT_l:
                if self.checkPointInNeighborhood(x_j_l, x_n_NOT_l, neighborhoodType="l2_ball"):
                    temp_set_holder += 1
            C_l_j_sets.append(temp_set_holder)
        C_l_j = [(self.lambda_ + C_l_j_set) for C_l_j_set in C_l_j_sets]
        data["C_l_j"] = C_l_j
        return(data)

    def checkCoverOfNeighborhood_2(self, label, neighborhoodType="l2_ball"):
        data = self.C_l_j_constructor_1(label=label, neighborhoodType="l2_ball")
        X_l = data["X_l"]
        size_l = data["size_l"]
        constraintMat = np.zeros(shape=(size_l,size_l))
        for x_n_l in range(size_l):
            for x_j_l in range(size_l):
                if self.checkPointInNeighborhood(X_l[x_j_l], X_l[x_n_l], neighborhoodType="l2_ball"):
                    constraintMat[x_n_l, x_j_l] = 1
        data["constraintMat"] = constraintMat
        return(data)

    def train_lp(self, verbose=False):
        '''Implement the linear programming formulation 
        and solve using cvxpy for prototype selection'''
        L = self.getLabels()
        alpha_j_l_lin_holder = np.zeros(shape=self.size)
        xi_i_l_lin_holder = np.zeros(shape=self.size)
        optimal_value_lin_holder = []
        data_dicts_holder = []
        for label in L:
            print("Working on label {}...".format(label))
            
            data = self.checkCoverOfNeighborhood_2(label=label, neighborhoodType="l2_ball")
            size_l = data["size_l"]
            C_l_j = data["C_l_j"]
            alpha_j_l = cvx.Variable(size_l,1)
            big1 = np.ones(shape=(size_l))
            xi_i_l = cvx.Variable(size_l,1)
            constraintMat = data["constraintMat"]

            objective = cvx.Minimize( (C_l_j * alpha_j_l) + (big1.T * xi_i_l))

            constraints = [constraintMat * alpha_j_l >= (big1 - xi_i_l), 
            alpha_j_l <=1, 
            alpha_j_l >=0,
            xi_i_l >=0]

            prob = cvx.Problem(objective, constraints)
            prob.solve()  # Returns the optimal value.
            data["alpha_j_l_lin"] = alpha_j_l.value
            data["xi_i_l_lin"] = xi_i_l.value
            data["optimal_value_lin"] = prob.value
            optimal_value_lin_holder.append(prob.value)
            data_dicts_holder.append(data)

            indices_l = data["indices_l"]
            for index in indices_l:
                alpha_j_l_lin_holder[index] = alpha_j_l.value[indices_l.index(index)]
                xi_i_l_lin_holder[index] = xi_i_l.value[indices_l.index(index)]

        self.alpha_j_l_lin_holder = alpha_j_l_lin_holder
        self.xi_i_l_lin_holder = xi_i_l_lin_holder
        self.optimal_value_lin = sum(optimal_value_lin_holder)
        self.data_dicts = data_dicts_holder
        print("linear program... solved!")

        
    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        
    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state=42)
    score = 0
    prots = 0
    for train_index, test_index in kf.split(data):
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
        '''implement code to count the total number of prototypes learnt and store it in prots'''
    score /= k    
    prots /= k
    obj_val /= k
    return score, prots, obj_val



def gmm_2d_data_maker(pi_array_of_mixing_weights, 
                      mu_array_of_means, 
                      R_array_of_covs, 
                      N_number_of_samples):
    exiter = []
    
    pi_array_of_mixing_weights = np.asarray(pi_array_of_mixing_weights)
    K_number_of_gaussians = pi_array_of_mixing_weights.shape[0]
    mu_array_of_means = np.asarray(mu_array_of_means)
    R_array_of_covs = np.asarray(R_array_of_covs)
    
    
    if len(pi_array_of_mixing_weights) != (K_number_of_gaussians):
        print("Update your number of Gaussians for the given amount of probabilities!")
        exiter.append(0)
        
    elif (np.sum(pi_array_of_mixing_weights)) != 1:
        print("Your probabilities don't sum to 1!")
        exiter.append(0)
        
    else:
        sample_holderx = np.zeros(shape=(N_number_of_samples))
        sample_holdery = np.zeros(shape=(N_number_of_samples))
        sample_holderl = np.zeros(shape=(N_number_of_samples), dtype=int)
        #plt.figure(figsize=(10,10), dpi=100)
        for i in range(N_number_of_samples):
            #print(i)
            k = int(np.random.choice(K_number_of_gaussians,p=pi_array_of_mixing_weights))

            x, y = np.random.multivariate_normal(mu_array_of_means[k],
                                                   R_array_of_covs[k])
            #plt.plot(x, y, marker='x', c='C{}'.format(k), alpha=pi_array_of_mixing_weights[k])
            sample_holderx[i]=x
            sample_holdery[i]=y
            sample_holderl[i]=k
        #plt.grid(b=True, which='major', alpha=0.25)
        #plt.title("Gaussian Mixture Model for 2 dimensions with {} Gaussians".format(K_number_of_gaussians))
        #plt.show()

        X = np.stack((sample_holderx,sample_holdery), axis=1)

        exiter = (X,sample_holderl)
   
    
    return(exiter)
    
TrainData_GMM = gmm_2d_data_maker([0.2,0.5,0.3],
                            [[15,0],[0,0],[0,15]],
                            [[[0.1,0],[0,0.1]],[[0.3,0],[0,0.3]],[[0.5,0],[0,0.5]]],
                            1000)

X_GMM = (TrainData_GMM)[0]
y_GMM = (TrainData_GMM)[1]
lambda_GMM = 1 / (X_GMM.shape)[0]

classifierGMM = classifier(X=X_GMM, y=y_GMM, epsilon_=5, lambda_=lambda_GMM)
print(classifierGMM.getLabels())
print(classifierGMM.getNumOfLabels())
data_1GMM = (classifierGMM.instantiate_dataDict_0(desiredLabel=1))
data_2GMM = (classifierGMM.instantiate_dataDict_0(desiredLabel=2))
print(classifierGMM.checkPointInNeighborhood(x0=data_1GMM["X_l"][0], x_test=data_1GMM["X_l"][1], neighborhoodType="l2_ball"))
print(classifierGMM.checkPointInNeighborhood(x0=data_1GMM["X_l"][0], x_test=data_2GMM["X_l"][0], neighborhoodType="l2_ball"))
classifierGMM.train_lp()
plt.figure(figsize=(8,8))
print(max(classifierGMM.alpha_j_l_lin_holder))
for i in range(classifierGMM.size):
    plt.plot(classifierGMM.X[i][0],classifierGMM.X[i][1], marker='x', c='C{}'.format(classifierGMM.y[i]), alpha=0.1)
    plt.plot(classifierGMM.X[i][0],classifierGMM.X[i][1], marker='o', markersize=10, c='C{}'.format(classifierGMM.y[i]), alpha=round(classifierGMM.alpha_j_l_lin_holder[i]))
plt.show()
plt.savefig('gmm.png')

TrainData_IRIS = load_iris(return_X_y=True)
X_IRIS = (TrainData_IRIS)[0]
y_IRIS = (TrainData_IRIS)[1]
lambda_IRIS = 1 / (X_IRIS.shape)[0]

classifierIRIS = classifier(X=X_IRIS, y=y_IRIS, epsilon_=1, lambda_=lambda_IRIS)
print(classifierIRIS.getLabels())
print(classifierIRIS.getNumOfLabels())
data_1IRIS = (classifierIRIS.instantiate_dataDict_0(desiredLabel=1))
data_2IRIS = (classifierIRIS.instantiate_dataDict_0(desiredLabel=2))
print(classifierIRIS.checkPointInNeighborhood(x0=data_1IRIS["X_l"][0], x_test=data_1IRIS["X_l"][1], neighborhoodType="l2_ball"))
print(classifierIRIS.checkPointInNeighborhood(x0=data_1IRIS["X_l"][0], x_test=data_2IRIS["X_l"][0], neighborhoodType="l2_ball"))




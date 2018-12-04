'''Libraries for Prototype selection'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxopt as cvx
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
        self.epsilon_ = epsilon_
        self.lambda_ = 1 / (self.X.shape)[0]
        

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

    def getSubsetOfDataGivenLabel(self, desiredLabel):
        '''Get a subset of the X data according to the label'''
        indices = [index for index, labels in enumerate(self.y) if labels == desiredLabel]
        X_l = self.X[indices]
        y_l = self.y[indices]
        size = (X_l.shape)[0]
        data = {
            "X_l": X_l,
            "y_l": y_l,
            "indices_l": indices,
            "size_l": size
        }
        return(data)

    def get_NOT_SubsetOfDataGivenLabel(self, NOT_desiredLabel):
        '''Get a subset of the X data that are not of the input class'''
        indices_NOT_l = [index for index, labels in enumerate(self.y) if labels != NOT_desiredLabel]
        X_NOT_l = self.X[indices_NOT_l]
        y_NOT_l = self.y[indices_NOT_l]
        size_NOT_l = (X_NOT_l.shape)[0]
        data_NOT_l = {
            "X_NOT_l": X_NOT_l,
            "y_NOT_l": y_NOT_l,
            "indices_NOT_l": indices_NOT_l,
            "size_NOT_l": size_NOT_l
        }
        return(data_NOT_l)

    def checkPointInNeighborhood(self, x0, x_test, neighborhoodType="l2_ball"):
        result = float('nan')
        if neighborhoodType == "l2_ball":
            result = (np.linalg.norm((x0-x_test), ord=2) <= self.epsilon_)

        return(result)

    def neighborhood_cap_NOT_SubsetOfDataGivenLabel(self, label, neighborhoodType="l2_ball"):
        data = getSubsetOfDataGivenLabel(desiredLabel=label)
        X_l = data["X_l"]
        data_NOT_l = get_NOT_SubsetOfDataGivenLabel(NOT_desiredLabel=label)
        X_NOT_l = data_NOT_l["X_NOT_l"]
        C_l_j_sets = []
        for x_j_l in X_l:
            temp_set_holder = {}
            for x_n_NOT_l in X_NOT_l:
                if checkPointInNeighborhood(x_j_l, x_n_NOT_l, neighborhoodType="l2_ball"):
                    temp_set_holder.add(x_n_NOT_l)
            C_l_j_sets.append(temp_set_holder)
        
        return(C_l_j_sets)

    def C_l_j_constructor(self, label, neighborhoodType="l2_ball"):
        C_l_j_sets = neighborhood_cap_NOT_SubsetOfDataGivenLabel(label=label, neighborhoodType="l2_ball")


    def train_lp(self, verbose=False):
        '''Implement the linear programming formulation 
        and solve using cvxpy for prototype selection'''
        
    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        
    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state = 42)
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
        sample_holderl = np.zeros(shape=(N_number_of_samples))
        plt.figure(figsize=(10,10), dpi=100)
        for i in range(N_number_of_samples):
            #print(i)
            k = (np.random.choice(K_number_of_gaussians,p=pi_array_of_mixing_weights))

            x, y = np.random.multivariate_normal(mu_array_of_means[k],
                                                   R_array_of_covs[k])
            #plt.plot(x, y, marker='x', c='C{}'.format(k), alpha=pi_array_of_mixing_weights[k])
            sample_holderx[i]=x
            sample_holdery[i]=y
            sample_holderl[i]=int(k)
        #plt.grid(b=True, which='major', alpha=0.25)
        #plt.title("Gaussian Mixture Model for 2 dimensions with {} Gaussians".format(K_number_of_gaussians))
        #plt.show()

        X = np.stack((sample_holderx,sample_holdery), axis=1)

        exiter = (X,sample_holderl)
   
    
    return(exiter)
    
TrainData_GMM = gmm_2d_data_maker([0.2,0.5,0.3],
                            [[5,0],[0,0],[0,5]],
                            [[[0.1,0],[0,0.1]],[[0.3,0],[0,0.3]],[[0.5,0],[0,0.5]]],
                            1000)

X_GMM = (TrainData_GMM)[0]
y_GMM = (TrainData_GMM)[1]
lambda_GMM = 1 / (X_GMM.shape)[0]

classiferGMM = classifier(X=X_GMM, y=y_GMM, epsilon_=1, lambda_=lambda_GMM)
print(classiferGMM.getLabels())
print(classiferGMM.getNumOfLabels())
data_1GMM = (classiferGMM.getSubsetOfDataGivenLabel(desiredLabel=1))
data_2GMM = (classiferGMM.getSubsetOfDataGivenLabel(desiredLabel=2))
print(classiferGMM.checkPointInNeighborhood(x0=data_1GMM["X_l"][0], x_test=data_1GMM["X_l"][1], neighborhoodType="l2_ball"))
print(classiferGMM.checkPointInNeighborhood(x0=data_1GMM["X_l"][0], x_test=data_2GMM["X_l"][0], neighborhoodType="l2_ball"))


TrainData_IRIS = load_iris(return_X_y=True)
X_IRIS = (TrainData_IRIS)[0]
y_IRIS = (TrainData_IRIS)[1]
lambda_IRIS = 1 / (X_IRIS.shape)[0]

classiferIRIS = classifier(X=X_IRIS, y=y_IRIS, epsilon_=1, lambda_=lambda_IRIS)
print(classiferIRIS.getLabels())
print(classiferIRIS.getNumOfLabels())
data_1IRIS = (classiferIRIS.getSubsetOfDataGivenLabel(desiredLabel=1))
data_2IRIS = (classiferIRIS.getSubsetOfDataGivenLabel(desiredLabel=2))
print(classiferIRIS.checkPointInNeighborhood(x0=data_1IRIS["X_l"][0], x_test=data_1IRIS["X_l"][1], neighborhoodType="l2_ball"))
print(classiferIRIS.checkPointInNeighborhood(x0=data_1IRIS["X_l"][0], x_test=data_2IRIS["X_l"][0], neighborhoodType="l2_ball"))




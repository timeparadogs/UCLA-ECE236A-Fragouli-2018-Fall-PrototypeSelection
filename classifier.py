'''Libraries for Prototype selection'''
import numpy as np
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
        

    '''Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose.'''

    def train_lp(self, verbose = False):
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
        plt.figure(figsize=(10,10), dpi=100)
        for i in range(N_number_of_samples):
            #print(i)
            k = np.random.choice(K_number_of_gaussians,p=pi_array_of_mixing_weights)

            x, y = np.random.multivariate_normal(mu_array_of_means[k],
                                                   R_array_of_covs[k])
            plt.plot(x, y, marker='x', c='C{}'.format(k), alpha=pi_array_of_mixing_weights[k])
            sample_holderx[i]=x
            sample_holdery[i]=y
        plt.grid(b=True, which='major', alpha=0.25)
        plt.title("Gaussian Mixture Model for 2 dimensions with {} Gaussians".format(K_number_of_gaussians))
        plt.show()
        
        exiter = np.stack((sample_holderx,sample_holdery)).T
   
    
    return(exiter)

    hw1spec = gmm_2d_data_maker([0.2,0.5,0.3],
                            [[20,0],[0,0],[0,20]],
                            [[[0.1,0],[0,0.1]],[[0.3,0],[0,0.3]],[[0.5,0],[0,0.5]]],
                            1000)
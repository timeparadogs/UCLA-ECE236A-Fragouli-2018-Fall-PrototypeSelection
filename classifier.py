'''Libraries for Prototype selection'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
from scipy.stats import bernoulli
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
        alpha_j_lin_holder = np.zeros(shape=self.size)
        xi_n_lin_holder = np.zeros(shape=self.size)
        optimal_value_lin_holder = []
        data_dicts_holder = []
        for label in L:
            if verbose:
                print("(train_lp) Working on label {}...".format(label))
            
            data = self.checkCoverOfNeighborhood_2(label=label, neighborhoodType="l2_ball")
            size_l = data["size_l"]
            C_l_j = data["C_l_j"]
            alpha_j_l = cvx.Variable(size_l,1)
            big1 = np.ones(shape=(size_l))
            big0 = np.zeros(shape=(size_l))
            xi_n_l = cvx.Variable(size_l,1)
            constraintMat = data["constraintMat"]
            if verbose:
                plt.imshow(constraintMat)
                plt.colorbar()
                plt.show()

            objective = cvx.Minimize( (C_l_j * alpha_j_l) + cvx.sum(xi_n_l))

            constraints = [constraintMat * alpha_j_l >= (big1 - xi_n_l), 
            alpha_j_l <= big1, 
            alpha_j_l <= 1,
            alpha_j_l >= big0,
            alpha_j_l >= 0,
            xi_n_l >= big0,
            xi_n_l >= big0,
            xi_n_l >= 0]

            prob = cvx.Problem(objective, constraints)
            prob.solve()  # Returns the optimal value.
            data["alpha_j_l_lin"] = [1 if alpha_j_l>=1 else alpha_j_l for alpha_j_l in alpha_j_l.value] 
            data["alpha_j_l_lin"] = [0 if alpha_j_l<=0 else alpha_j_l for alpha_j_l in data["alpha_j_l_lin"]] 
            data["xi_n_l_lin"] = [1 if xi_n_l>=1 else xi_n_l for xi_n_l in xi_n_l.value] 
            data["xi_n_l_lin"] = [0 if xi_n_l<=0 else xi_n_l for xi_n_l in data["xi_n_l_lin"]]           
            data["optimal_value_l_lin"] = prob.value
            optimal_value_lin_holder.append(prob.value)
            data_dicts_holder.append(data)

            indices_l = data["indices_l"]
            for index in indices_l:
                alpha_j_lin_holder[index] = alpha_j_l.value[indices_l.index(index)]
                xi_n_lin_holder[index] = xi_n_l.value[indices_l.index(index)]

        self.alpha_j_lin_holder = alpha_j_lin_holder
        self.xi_n_lin_holder = xi_n_lin_holder
        self.optimal_value_lin = sum(optimal_value_lin_holder)
        self.data_dicts = data_dicts_holder
        print("linear program... solved!")

        
    def objective_value(self, verbose=False):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''

        data_dicts = self.data_dicts

        L = self.getLabels()
        A_j_round_holder = np.zeros(shape=self.size)
        S_n_round_holder = np.zeros(shape=self.size)
        optimal_value_round_holder = []
        data_dicts_holder = []
        for label in L:
            if verbose:
                print("(objective_value) Working on label {}...".format(label))
            data = data_dicts[label]
            size_l = data["size_l"]
            A_j_l_round_holder = np.zeros(shape=size_l)
            S_n_l_round_holder = np.zeros(shape=size_l)
            OPT_ROUND_COMP = float('nan')
            C_l_j = data["C_l_j"]
            alpha_j_l_lin = data["alpha_j_l_lin"]
            xi_n_l_lin = data["xi_n_l_lin"]
            optimal_value_l_lin = data["optimal_value_l_lin"]
            OPT_LIN_COMP = 2* np.log(size_l)*optimal_value_l_lin
            redo = True
            feasibility_count = 0
            while redo:
                if verbose:
                    print("redo session: {}".format(feasibility_count))
                feasibility_count +=1
                for t in range( int(np.ceil(2 * np.log(size_l))) ):
                    for i in range(size_l):
                        A_j_l_temp_i = bernoulli.rvs(alpha_j_l_lin[i])
                        A_j_l_round_holder[i] = max(A_j_l_temp_i, A_j_l_round_holder[i])
                        S_n_l_temp_i = bernoulli.rvs(xi_n_l_lin[i])
                        S_n_l_round_holder[i] = max(S_n_l_temp_i, S_n_l_round_holder[i])
                
                constraintMat = data["constraintMat"]

                big1 = np.ones(shape=(size_l))

                firstTemp = sum([c_l_j*A_j_l for c_l_j,A_j_l in zip(C_l_j,A_j_l_round_holder)])
                secondTemp = sum(S_n_l_round_holder)
                OPT_ROUND_COMP = firstTemp + secondTemp

                LHS = constraintMat @ A_j_l_round_holder
                RHS = big1 - S_n_l_round_holder

                if verbose:
                    if all([lhs>=rhs for lhs,rhs in zip(LHS,RHS)]):
                        print("1st constraint passed: covering points in a label")
                        if all([A_j_l <= 1 for A_j_l in A_j_l_round_holder]):
                            print("2nd constraint passed: A_j_l <=1")
                            if all([A_j_l >= 0 for A_j_l in A_j_l_round_holder]):
                                print("3rd constraint passed: A_j_l >=0")
                                if all([(A_j_l == 0 or A_j_l == 1) for A_j_l in A_j_l_round_holder]):
                                    if all([S_n_l >= 0 for S_n_l in S_n_l_round_holder]):
                                        print("4th constraint passed: S_n_l >=0")
                                        if (OPT_ROUND_COMP <= OPT_LIN_COMP):
                                            print("objective value comparison passed")
                                            redo = False
                                        else:
                                            print(len(A_j_l_round_holder))
                                            print(sum(A_j_l_round_holder))
                                            print(len(S_n_l_round_holder))
                                            print(sum(S_n_l_round_holder))

                if all([lhs>=rhs for lhs,rhs in zip(LHS,RHS)]):
                    if all([A_j_l <= 1 for A_j_l in A_j_l_round_holder]):
                        if all([A_j_l >= 0 for A_j_l in A_j_l_round_holder]):
                            if all([(A_j_l == 0 or A_j_l == 1) for A_j_l in A_j_l_round_holder]):
                                if all([S_n_l >= 0 for S_n_l in S_n_l_round_holder]):
                                    if (OPT_ROUND_COMP <= OPT_LIN_COMP):
                                        redo = False


            data["A_j_l_round_holder"] = A_j_l_round_holder
            data["S_n_l_round_holder"] = S_n_l_round_holder
            data["optimal_value_l_round"] = OPT_ROUND_COMP
            optimal_value_round_holder.append(OPT_ROUND_COMP)
            data_dicts_holder.append(data)

            indices_l = data["indices_l"]
            for index in indices_l:
                A_j_round_holder[index] = A_j_l_round_holder[indices_l.index(index)]
                S_n_round_holder[index] = S_n_l_round_holder[indices_l.index(index)]
        
        self.A_j_round_holder = A_j_round_holder
        self.S_n_round_holder = S_n_round_holder
        self.optimal_value_round = sum(optimal_value_round_holder)
        self.X_proto = [x for x,A_j in zip(self.X,A_j_round_holder) if A_j == 1]
        self.X_proto = np.array(self.X_proto)
        self.y_proto = [y for y,A_j in zip(self.y,A_j_round_holder) if A_j == 1]
        self.size_proto = (self.X_proto.shape)[0]
        self.data_dicts = data_dicts_holder
        print("randomized rounding... solved!")
        return(self.optimal_value_round)
        
    def predict(self, X_instances):
        '''
        Predicts the label for an array of instances using the framework learnt
        and returns it along the same index
        '''
        X_proto = self.X_proto
        y_proto = self.y_proto

        y_instances = []

        for x_instance in X_instances:
            dist_holder = []
            for x_proto, in X_proto:
                dist_holder.append( np.linalg.norm((x_instance-x_proto), ord=2) )
            idx = np.argmin(dist_holder)
            y_instances.append(y_proto[idx])
        
        return(y_instances)

def cross_val(X, y, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state=42)
    score = 0
    prots = 0
    for train_index, test_index in kf.split(X):
        ps = classifier(X[train_index], y[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        score += sklearn.metrics.accuracy_score(y[test_index], ps.predict(X[test_index]))
        prots += ps.size_proto
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
                            [[0,0],[10,0],[20,0]],
                            [[[0.1,0],[0,10]],[[0.3,0],[0,11]],[[0.5,0],[0,12]]],
                            50)

X_GMM = (TrainData_GMM)[0]
y_GMM = (TrainData_GMM)[1]
lambda_GMM = 1 / (X_GMM.shape)[0]

count = 1
for epislon in range(6):
    classifierGMM = classifier(X=X_GMM, y=y_GMM, epsilon_=epislon, lambda_=lambda_GMM)
    classifierGMM.train_lp(verbose=False)
    classifierGMM.objective_value(verbose=False)
    plt.figure(figsize=(8,8))
    for i in range(classifierGMM.size):
        plt.plot(classifierGMM.X[i][0],classifierGMM.X[i][1], marker='x', c='C{}'.format(classifierGMM.y[i]), alpha=0.5)
    angle = np.linspace(0, 2*np.pi, num=100)
    for i in range(classifierGMM.size_proto):
        plt.plot(classifierGMM.X_proto[i][0],classifierGMM.X_proto[i][1], marker='o', c='C{}'.format(classifierGMM.y_proto[i]), alpha=1)
        plt.fill(classifierGMM.X_proto[i][0] + classifierGMM.epsilon_*np.cos(angle), classifierGMM.X_proto[i][1] + classifierGMM.epsilon_*np.sin(angle), c='C{}'.format(classifierGMM.y_proto[i]), alpha=0.0625)
    plt.axis('equal')
    plt.title('epsilon= {}'.format(epislon))
    #plt.savefig('gifFolder/gmm{}.png'.format(count))
    count+=1
    plt.show()


TrainData_IRIS = load_iris(return_X_y=True)
X_IRIS = (TrainData_IRIS)[0]
y_IRIS = (TrainData_IRIS)[1]
lambda_IRIS = 1 / (X_IRIS.shape)[0]

for epislon in range(7):
    classifierIRIS = classifier(X=X_IRIS, y=y_IRIS, epsilon_=1, lambda_=lambda_IRIS)
    classifierIRIS.train_lp(verbose=True)
    classifierIRIS.objective_value(verbose=True)




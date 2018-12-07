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
from scipy.spatial.distance import pdist, squareform

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
            #if verbose:
                #plt.imshow(constraintMat)
                #plt.colorbar()
                #plt.show()

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
            prob.solve(solver=cvx.SCS)  # Returns the optimal value.
            data["alpha_j_l_lin"] = [1 if alpha_j_l>=1 else alpha_j_l for alpha_j_l in alpha_j_l.value] 
            data["alpha_j_l_lin"] = [0 if alpha_j_l<=0 else alpha_j_l for alpha_j_l in data["alpha_j_l_lin"]] 
            data["xi_n_l_lin"] = [1 if xi_n_l>=1 else xi_n_l for xi_n_l in xi_n_l.value] 
            data["xi_n_l_lin"] = [0 if xi_n_l<=0 else xi_n_l for xi_n_l in data["xi_n_l_lin"]]

            firstTemp = sum([c_l_j*alpha_j_l for c_l_j,alpha_j_l in zip(C_l_j,data["alpha_j_l_lin"] )])
            secondTemp = sum(data["xi_n_l_lin"])
            OPT_LIN_l = firstTemp + secondTemp

            data["optimal_value_l_lin"] = OPT_LIN_l
            if verbose:
                print("optimal value for class {l}: {p}".format(l=label, p=OPT_LIN_l))
                print(prob.value)
            optimal_value_lin_holder.append(OPT_LIN_l)
            data_dicts_holder.append(data)

            indices_l = data["indices_l"]
            for index in indices_l:
                alpha_j_lin_holder[index] = alpha_j_l.value[indices_l.index(index)]
                xi_n_lin_holder[index] = xi_n_l.value[indices_l.index(index)]

        self.alpha_j_lin_holder = alpha_j_lin_holder
        self.xi_n_lin_holder = xi_n_lin_holder
        self.optimal_value_lin = sum(optimal_value_lin_holder)
        if verbose:
            print("optimal value: {p}".format(p=sum(optimal_value_lin_holder)))
        self.data_dicts = data_dicts_holder
        if verbose:
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
            A_j_l_round_holder = np.zeros(shape=size_l, dtype=int)
            S_n_l_round_holder = np.zeros(shape=size_l, dtype=int)
            OPT_ROUND_COMP = float('nan')
            C_l_j = data["C_l_j"]
            alpha_j_l_lin = data["alpha_j_l_lin"]
            xi_n_l_lin = data["xi_n_l_lin"]
            optimal_value_l_lin = data["optimal_value_l_lin"]
            OPT_LIN_COMP = 2* np.log(size_l) * optimal_value_l_lin
            redo = True
            feasibility_count = 0
            while redo:
                if verbose:
                    print("redo session: {}".format(feasibility_count))
                feasibility_count +=1
                if feasibility_count > 30:
                    print('check randomized rounding...')
                A_j_l_round_holder = np.zeros(shape=size_l, dtype=int)
                S_n_l_round_holder = np.zeros(shape=size_l, dtype=int)
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
                                            print(OPT_LIN_COMP)
                                            print(OPT_ROUND_COMP)
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
        if verbose:
            print("randomized rounding... solved!")
        return(self.optimal_value_round)
        
    def predict(self, X_instances):
        '''
        Predicts the label for an array of instances using the framework learnt
        and returns it along the same index
        '''
        X_proto = self.X_proto
        y_proto = self.y_proto

        y_instances_test = []
        y_instances_cover = []

        for x_instance in X_instances:
            dist_holder = []
            neighborhood_checker = []
            for x_proto in X_proto:
                dist = np.linalg.norm((x_instance-x_proto), ord=2)
                dist_holder.append( dist )
                neighborhood_checker.append( dist <= self.epsilon_ )

            idx = np.argmin(dist_holder)
            y_instances_test.append(y_proto[idx])

            if sum(neighborhood_checker) == 1:
                idx = neighborhood_checker.index(1)
                y_instances_cover.append(y_proto[idx])
            else:
                y_instances_cover.append(-50)

        return(y_instances_test, y_instances_cover)

def cross_val(X, y, epsilon_, lambda_, k, verbose=False):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state=42)
    score = 0
    prots = 0
    obj_val = 0
    test_score = 0
    test_error = 0
    cover_score = 0
    cover_error = 0
    for train_index, test_index in kf.split(X):
        ps = classifier(X[train_index], y[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value(verbose)
        test_score += sklearn.metrics.accuracy_score(y[test_index], ps.predict(X[test_index])[0])
        test_error += (1 - sklearn.metrics.accuracy_score(y[test_index], ps.predict(X[test_index])[0]))
        cover_score += sklearn.metrics.accuracy_score(y[test_index], ps.predict(X[test_index])[1])
        cover_error += (1 - sklearn.metrics.accuracy_score(y[test_index], ps.predict(X[test_index])[1]))
        prots += ps.size_proto
        if verbose:
            print("finished with a fold!")
    test_score /= k  
    test_error /= k
    cover_score /= k
    cover_error /= k
    prots /= k
    obj_val /= k
    return test_score, test_error, cover_score, cover_error, prots, obj_val

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

# explicitly calculate the whole n x n distance matrix
dist_mat_GMM = squareform(pdist(X_GMM, metric="euclidean"))
dist_mat_GMM = dist_mat_GMM.flatten()
dist_mat_GMM = [distance for distance in dist_mat_GMM if distance > 0]
percentile2_GMM = np.percentile(dist_mat_GMM, q=2)
percentile40_GMM = np.percentile(dist_mat_GMM, q=40)
percentile60_GMM = np.percentile(dist_mat_GMM, q=60)
epsilon_range_GMM = np.linspace(start=percentile2_GMM, stop=percentile60_GMM, num=500)

test_error_holder_GMM = []
cover_error_holder_GMM = []
prot_number_holder_GMM = []
for epsilon in epsilon_range_GMM:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_GMM, y_GMM, epsilon, lambda_GMM, k=4, verbose=False)
    test_error_holder_GMM.append(test_error)
    cover_error_holder_GMM.append(cover_error)
    prot_number_holder_GMM.append(prots)

plt.figure(figsize=(8.5,5.5))
plt.scatter(prot_number_holder_GMM, test_error_holder_GMM, c='r', marker='o', alpha=0.5)
plt.title('Test Error on the Gaussian Mixtures Dataset')
plt.ylabel('Test Error')
plt.xlabel('Average Number of Prototypes')
plt.show()

plt.figure(figsize=(8.5,5.5))
plt.scatter(prot_number_holder_GMM, cover_error_holder_GMM, c='b', marker='o', alpha=0.5)
plt.title('Cover Error on the Gaussian Mixtures Dataset')
plt.ylabel('Cover Error')
plt.xlabel('Average Number of Prototypes')
plt.show()

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
epsilon_range_IRIS = np.linspace(start=percentile2_IRIS, stop=percentile60_IRIS, num=5000)

test_error_holder_IRIS = []
cover_error_holder_IRIS = []
prot_number_holder_IRIS = []
for epsilon in epsilon_range_IRIS:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_IRIS, y_IRIS, epsilon, lambda_IRIS, k=4, verbose=False)
    test_error_holder_IRIS.append(test_error)
    cover_error_holder_IRIS.append(cover_error)
    prot_number_holder_IRIS.append(prots)

plt.figure(figsize=(8.5,5.5))
plt.scatter(prot_number_holder_IRIS, test_error_holder_IRIS, c='r', marker='o', alpha=0.5)
plt.title('Test Error on the Iris Dataset')
plt.ylabel('Test Error')
plt.xlabel('Average Number of Prototypes')
plt.show()

plt.figure(figsize=(8.5,5.5))
plt.scatter(prot_number_holder_IRIS, cover_error_holder_IRIS, c='b', marker='o', alpha=0.5)
plt.title('Cover Error on the Iris Dataset')
plt.ylabel('Cover Error')
plt.xlabel('Average Number of Prototypes')
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
epsilon_range_CANC = np.linspace(start=percentile2_CANC, stop=percentile40_CANC, num=50)

test_error_holder_CANC = []
cover_error_holder_CANC = []
prot_number_holder_CANC = []
for epsilon in epsilon_range_CANC:
    print("on epsilon = {}".format(round(epsilon,4)))
    test_score, test_error, cover_score, cover_error, prots, obj_val = cross_val(X_CANC, y_CANC, epsilon, lambda_CANC, k=4, verbose=False)
    test_error_holder_CANC.append(test_error)
    cover_error_holder_CANC.append(cover_error)
    prot_number_holder_CANC.append(prots)

plt.figure(figsize=(8.5,5.5))
plt.scatter(prot_number_holder_CANC, test_error_holder_CANC, c='r', marker='o', alpha=0.75)
plt.title('Test Error on the Breast Cancer Dataset')
plt.ylabel('Test Error')
plt.xlabel('Average Number of Prototypes')
plt.show()

plt.figure(figsize=(8.5,5.5))
plt.plot(prot_number_holder_CANC, cover_error_holder_CANC, c='r', marker='o', alpha=0.75)
plt.title('Cover Error on the Breast Cancer Dataset')
plt.ylabel('Cover Error')
plt.xlabel('Average Number of Prototypes')
plt.show()
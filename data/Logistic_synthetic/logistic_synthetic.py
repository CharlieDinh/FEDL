import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math


NUM_USER = 30

def logit(X, w):
    return 1 / (1 + np.exp(-X.dot(w)))

def generate_synthetic( kappa = 10, dimension = 100):
    if kappa == 1:
        LAMBDA = 100
    else:
        LAMBDA = 1 / (kappa - 1)
    #L = 1
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    print(samples_per_user)
    #num_samples = np.sum(samples_per_user)

    noise_ratio = 0.01
    kappa = 10

    X = [[] for _ in range(NUM_USER)]
    Y = [[] for _ in range(NUM_USER)]

    if(kappa == 1):
        LAMBDA = 100
    else:
        LAMBDA = 1 / (kappa - 1)
    for i in range(NUM_USER):
        xx = np.random.randn(samples_per_user[i], dimension)
        X[i] = xx.tolist()
    X = np.array(X)
    max_norm = 0
    for i in range(NUM_USER):
        X[i] = np.array(X[i])
        max_norm = max(np.sqrt(np.linalg.norm(np.array(X[i]).T.dot(np.array(X[i])), 2) / samples_per_user[i]),max_norm)


    X = np.asarray(X) / (max_norm + LAMBDA)
    w_0 = np.random.rand(dimension)


    for i in range(NUM_USER):
        Y_0 = logit(X[i], w_0)
        #if(Y_0 <= 0.5):
        print(Y_0)
        X[i] = X[i].tolist()
        Y_0[Y_0 > 0.5] = 1
        Y_0[Y_0 <= 0.5] = 0
        noise = np.random.binomial(1, noise_ratio, samples_per_user[i])
        yy = np.multiply(noise - Y_0, noise) + np.multiply(Y_0, 1 - noise)
        Y[i] = yy.tolist()

    print("{}-th users has {} exampls".format(i, len(Y[i])))

    X = X.tolist()
    print(type(X),type(Y))
    return X, Y



def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = "./data//train/mytrain.json"
    test_path = "./data//test/mytest.json"

    X, y = generate_synthetic(10,100)     # synthetiv (0,0)

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()


# --- New data generation method ---
def generate_synthetic_non_iid(alpha, beta, p=1.2, NUM_USERS=100,
                               DIMENSION=60, NUM_CLASSES=10, kappa=10):
    """
    Generate non-iid synthetic data for classification.
    :return: X and y for each user
    :param alpha: stddev of u
    :param beta: stddev of B
    :param p: parameter in diagonal matrix (controlling kappa)
    :param NUM_USERS: total number of users
    :param DIMENSION: dimension of data points
    :param NUM_CLASSES: number of output classes
    :return:
    """

    # For consistent outcomes
    np.random.seed(0)

    OUTPUT_DIM = 1 if NUM_CLASSES == 2 else NUM_CLASSES  # Determine if logistic regression
    SAMPLES_PER_USER = np.random.lognormal(4, 2, NUM_USERS).astype(int) + 50
    NUM_TOTAL_SAMPLES = np.sum(SAMPLES_PER_USER)

    X_split = [[] for _ in range(NUM_USERS)]   # X for each user
    y_split = [[] for _ in range(NUM_USERS)]   # y for each user

    u = np.random.normal(0, alpha, NUM_USERS)  # u_k = mean of W_k
    B = np.random.normal(0, beta, NUM_USERS)   # B_k = mean of v_k

    # Find v (mean of X)
    v = np.zeros((NUM_USERS, DIMENSION))
    for k in range(NUM_USERS):
        v[k] = np.random.normal(B[k], 1, DIMENSION)

    # Covariance matrix
    diagonal = np.array([(j+1) ** -p for j in range(DIMENSION)])
    Sigma = np.diag(diagonal)

    def softmax(x):
        exp = np.exp(x)
        return exp / np.sum(exp)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Generate X for each user
    LAMBDA = 100 if kappa == 1 else (1 / (kappa - 1))
    max_norm = 0
    for k in range(NUM_USERS):
        X_k = np.random.multivariate_normal(v[k], Sigma, SAMPLES_PER_USER[k])
        max_norm = max(max_norm,
                       np.linalg.norm(X_k.T.dot(X_k), 2) / SAMPLES_PER_USER[k])
        X_split[k] = X_k
    # Normalize X for each user using max_norm and LAMBDA
    for k in range(NUM_USERS):
        X_split[k] /= max_norm + LAMBDA

    # Generate y for each user
    for k in range(NUM_USERS):
        # W_k ~ N(u_k, 1) (In Network-DANE W_k is generated uniformly randomly)
        W_k = np.random.normal(u[k], 1, (DIMENSION, OUTPUT_DIM))

        # b_k ~ N(u_k, 1) (In Network-DANE there is no bias)
        b_k = np.random.normal(u[k], 1, OUTPUT_DIM)

        X_k = X_split[k]

        y_k = np.zeros(SAMPLES_PER_USER[k])
        for i in range(SAMPLES_PER_USER[k]):
            if NUM_CLASSES == 2:
                # Logistic regression
                # (In Network-DANE y_k = sigmoid(W_k * x_k + noise)
                # where noise ~ N(0, I))
                y_k[i] = int(sigmoid(np.dot(X_k[i], W_k) + b_k) > 0.5)
            else:
                # Multinomial regression
                y_k[i] = np.argmax(softmax(np.dot(X_k[i], W_k) + b_k))

        X_split[k] = X_split[k].tolist()
        y_split[k] = y_k.tolist()

        print("User {} has {} data points.".format(k, y_k.shape[0]))

    print("Total number of samples: {}".format(NUM_TOTAL_SAMPLES))
    return X_split, y_split

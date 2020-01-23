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


import numpy as np
import os
import json
import random


np.random.seed(0)


def generate_linear_regression_data(NUM_USERS=100, DIMENSION=200, kappa=10, noise_variance=0.1):
    SAMPLES_PER_USER = np.random.lognormal(4, 2, NUM_USERS).astype(int) + 50
    NUM_TOTAL_SAMPLES = np.sum(SAMPLES_PER_USER)

    X_split = [[] for _ in range(NUM_USERS)]  # X for each user
    y_split = [[] for _ in range(NUM_USERS)]  # y for each user

    powers = - np.log(kappa) / np.log(DIMENSION) / 2

    for k in range(NUM_USERS):
        S = np.power(np.arange(DIMENSION) + 1, powers)
        X = np.random.rand(SAMPLES_PER_USER[k], DIMENSION)
        X *= S
        X /= np.linalg.norm(X.T.dot(X), 2) / SAMPLES_PER_USER[k]

        # L = 1, \beta = 1 / kappa
        W = np.random.rand(DIMENSION)
        
        y = np.dot(X, W)
        # Add some noise to y
        np.sqrt(noise_variance) * np.random.rand(SAMPLES_PER_USER[k])


        X_split[k] = X.tolist()
        y_split[k] = y.tolist()

        print("User {} has {} data points.".format(k, SAMPLES_PER_USER[k]))

    print("Total number of samples: {}".format(NUM_TOTAL_SAMPLES))

    return X_split, y_split


def main():
    NUM_USERS = 100
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    if not os.path.exists("data/linear_regression/train"):
        os.makedirs("data/linear_regression/train")
    if not os.path.exists("data/linear_regression/test"):
        os.makedirs("data/linear_regression/test")

    train_path = "data/linear_regression/train/mytrain.json"
    test_path = "data/linear_regression/test/mytest.json"

    X, y = generate_linear_regression_data(NUM_USERS=NUM_USERS, DIMENSION=200,
                                           kappa=10, noise_variance=0.1)


    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

        
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

    
    # --- Uncomment the code below to save data for each user in separate files ---
    # train_path = os.path.join("data", "linear_regression", "userstrain")
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    # 
    # test_path = os.path.join("data", "linear_regression", "userstest")
    # if not os.path.exists(test_path):
    #     os.makedirs(test_path)
    # 
    # with open(os.path.join("data", "linear_regression", "train", "mytrain.json"), "r") as f_train:
    #     train = json.load(f_train)
    # for i in range(NUM_USERS):
    #     data = {}
    #     data['id'] = train['users'][i]
    #     data['X'] = train["user_data"][data['id']]['x']
    #     data['y'] = train["user_data"][data['id']]['y']
    #     data['num_samples'] = train["num_samples"][i]
    #     with open(os.path.join("data/linear_regression/userstrain", data['id'] + ".json"), "w") as f:
    #         json.dump(data, f)
    # 
    # 
    # with open(os.path.join("data", "linear_regression", "test", "mytest.json"), "r") as f_test:
    #     test = json.load(f_test)
    # for i in range(100):
    #     data = {}
    #     data['id'] = test['users'][i]
    #     data['X'] = test["user_data"][data['id']]['x']
    #     data['y'] = test["user_data"][data['id']]['y']
    #     data['num_samples'] = test["num_samples"][i]
    #     with open(os.path.join("data/linear_regression/userstest", data['id'] + ".json"), "w") as f:
    #         json.dump(data, f)
if __name__ == '__main__':
    main()
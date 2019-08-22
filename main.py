import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py


# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'fedsvrg', 'fedsarah', 'fedsgd']

DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1', 'fashion_mnist']  # NIST is EMNIST in the paper

DATA_SET = "mnist"

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,),  # num_classes
    'sent140.stacked_lstm': (25, 2, 100),  # seq_len, num_classes, num_hidden
    # seq_len, num_classes, num_hidden
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100),
    # num_classes, should be changed to 62 when using EMNIST
    'nist.mclr': (62,),
    'nist.cnn': (62,),
    'mnist.mclr': (10,),  # num_classes
    'mnist.cnn': (10,),  # num_classes
    'fashion_mnist.mclr':(10,),
    'fashion_mnist.cnn': (10,),
    'shakespeare.stacked_lstm': (80, 80, 256),  # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, )  # num_classes
}


def read_options(num_users=5, loc_ep=10, Numb_Glob_Iters=100, lamb=0, learning_rate=0.01, alg='fedprox', weight=True):
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default=alg)  # fedavg, fedprox
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default=DATA_SET)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr.py')  # 'stacked_lstm.py'
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=Numb_Glob_Iters)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=num_users)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=1
                        )  # 0 is full dataset
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=loc_ep)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=learning_rate)  # 0.003
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0.)  # 0.01
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--weight',
                        help='enable weight value;',
                        type=int,
                        default=weight)
    parser.add_argument('--lamb',
                        help='Penalty value for proximal term;',
                        type=int,
                        default=lamb)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # load selected model
    # all synthetic datasets use the same model
    if parsed['dataset'].startswith("synthetic"):
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % (
            'flearn', 'models', parsed['dataset'], parsed['model'])

    # mod = importlib.import_module(model_path)
    import flearn.models.mnist.mclr as mclr
    mod = mclr
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(
        model_path.split('.')[2:-1])]
    # parsed['model_params'] = MODEL_PARAMS['mnist.mclr']

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    return parsed, learner, optimizer


def main(num_users=5, loc_ep=10, Numb_Glob_Iters=100, lamb=0, learning_rate=0.01, alg='fedprox', weight=True):
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # parse command line arguments
    options, learner, optimizer = read_options(
        num_users, loc_ep, Numb_Glob_Iters, lamb, learning_rate, alg, weight)

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()


def simple_read_data(loc_ep, alg):
    hf = h5py.File('{}_{}.h5'.format(alg, loc_ep), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc


def plot_summary(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[]):

    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + str(learning_rate[i])
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], DATA_SET + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:],linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(train_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='best')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    plt.savefig('train_acc.png')

    plt.figure(2)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='best')
    #plt.ylim([0, 0.3])
    plt.ylabel('Training Loss')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    #plt.ylim([train_loss.min(), 0.3])
    plt.savefig('train_loss.png')

    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='best')
    #plt.ylim([0.9, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    plt.savefig('glob_acc.png')


if __name__ == '__main__':
    algorithms_list = ["fedsarah", "fedsgd", "fedsvrg"]
    lamb_value = [0,0,0]
    learning_rate = [0.01, 0.01, 0.01]
    local_ep = [20,20,20]
    if(0):
        plot_summary(num_users=100, loc_ep1=50, Numb_Glob_Iters=100,
                     lamb=lamb_value, learning_rate=learning_rate, algorithms_list=algorithms_list)
    else:
        for i in range(len(algorithms_list)):
            main(num_users=10, loc_ep=local_ep[i], Numb_Glob_Iters = 200, lamb=lamb_value[i], learning_rate=learning_rate[i], alg=algorithms_list[i])

        plot_summary(num_users=10, loc_ep1=local_ep, Numb_Glob_Iters = 200, lamb=lamb_value, learning_rate=learning_rate, algorithms_list=algorithms_list)

        print("-- FINISH -- :",)

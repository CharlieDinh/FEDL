import matplotlib.pyplot as plt
import h5py
import numpy as np

def simple_read_data(loc_ep, alg):
    hf = h5py.File("./"+'{}_{}.h5'.format(alg, loc_ep), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset = ""):
    
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + \
            "_" + str(learning_rate[i]) + "_" + str(num_users) + \
            "u" + "_" + str(batch_size[i]) + "b"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)
    MIN = train_loss.min() - 0.001
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(train_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig('train_acc.png')

    plt.figure(2)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    #plt.ylim([MIN, 1])
    plt.ylabel('Training Loss')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    #plt.ylim([train_loss.min(), 1])
    plt.savefig('train_loss.png')

    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algs_lbl[i])
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Number of Global Iterations')
    plt.title('Number of users: ' + str(num_users) +
              ', Lr: ' + str(learning_rate[0]))
    plt.savefig('glob_acc.png')

def plot_summary_two_figures(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size = 0, dataset = ""):
    
    #+'$\mu$'

    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + \
            str(learning_rate[i]) + "_" + str(num_users) + "u"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)
    linestyles = ['-', '--', '-.', ':', '--', '-.']
    algs_lbl = ["FedProxVR_Sarah", "FedProxVR_Svrg", "FedAvg", "FedProx",
                "FedProxVR_Sarah", "FedProxVR_Svrg", "FedAvg", "FedProx"]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #min = train_loss.min()
    min = train_loss.min() - 0.001
    num_al = 4
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    for i in range(num_al):
        ax2.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax2.set_ylim([min, 0.34])
        ax2.legend(loc='upper right')
        ax2.set_title("MNIST: 100 users, " +
                      r'$\beta =7,$' + r'$\tau = 20$', y=1.02)

    for i in range(num_al):
        ax1.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_ylim([min, 0.34])
        ax1.legend(loc='upper right')
        ax1.set_title("MNIST: 100 users, " +
                      r'$\beta = 5,$' + r'$\tau = 10$', y=1.02)

    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig('train_loss.pdf')

    plt.figure(2)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = glob_acc.max() + 0.01
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    for i in range(num_al):
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax2.set_ylim([0.8, max])
        ax2.legend(loc='upper right')
        ax2.set_title("MNIST: 100 users, " +
                      r'$\beta = 10,$' + r'$\tau = 50$', y=1.02)

    for (i) in range(num_al):
        ax1.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_title("MNIST: 100 users, " +
                      r'$\beta = 7,$' + r'$\tau = 20$', y=1.02)
        ax1.set_ylim([0.8, max])
        ax1.legend(loc='upper right')
    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Test Accuracy', labelpad=15)
    plt.savefig('glob_acc.pdf')
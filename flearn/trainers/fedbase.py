import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
import h5py

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        self.parameters = params
        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.rho = self.parameters['rho']
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self, prox=False, lamb=0, learning_rate=0, data_set="", num_users=0, batch=0):
        alg = data_set + self.parameters['optimizer']

        if(prox == True):
            alg = alg + "_prox_" + str(lamb)
        alg = alg + "_" + str(learning_rate) + "_" + str(num_users) + "u" + "_" + str(self.batch_size) + "b"
        if(self.rho > 0):
            alg += "_" + str(self.rho) + "r"
        endstr = str(self.parameters['num_epochs']) + "_" +  str(self.index)
        with h5py.File("./results/"+'{}_{}.h5'.format(alg, endstr), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
            hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.close()
        # pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_clients == len(self.clients)):
            print("All users are selected")
            return self.clients

        num_clients = min(num_clients, len(self.clients))
        #np.random.seed(round + self.index) 
        np.random.seed(round * (self.index + 1))
        return np.random.choice(self.clients, num_clients, replace=False) #, p=pk)


    def aggregate(self, wsolns, weighted=True):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of samples
            # Equal weights
#            if(weighted==False):
#                w=1 # Equal weights
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]
        return averaged_soln

    def aggregate_derivate(self, fsolns, weighted=True):
        total_derivative = 0.0
        base = [0]*len(fsolns[0][1])
        for (f, soln) in fsolns:  # w is the number of samples
            total_derivative += f
            for i, v in enumerate(soln):
                base[i] += f*v.astype(np.float64)

        averaged_soln = [v / total_derivative for v in base]
        return averaged_soln



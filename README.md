# Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation

This repository is for the Experiment Section of the paper:
"Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation"

Authors:
Canh T. Dinh, Nguyen H. Tran, Minh N. H. Nguyen, Choong Seon Hong, Wei Bao, Albert Zomaya, Vincent Gramoli

Link:
https://arxiv.org/abs/1910.13067

Our Code is developed based on the code from: 
https://github.com/litian96/FedProx

# Software requirements:
- numpy, scipy, tensorflow, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 3 datasets: MNIST, FEMNIST, and Synthetic 

- To generate non-idd MNIST Data: In folder data/mnist,  run: "python3 generate_niid_mnist_100users.py" 
- To generate FENIST Data: first In folder data/nist run preprocess.sh to obtain all raw data, or can be download in the link below, then run  python3 generate_niid_femnist_100users.py
- To generate niid Linear Synthetic: In folder data/linear_synthetic, run: "python3 generate_linear_regession.py" 
- The datasets are available to download at: https://drive.google.com/drive/folders/1Q91NCGcpHQjB3bXJTvtx5qZ-TrIZ9WzT?usp=sharing

# Produce figures in the paper:
- There is a main file "main.py" which allows running all experiments and 3 files "main_mnist.py, main_nist.py, main_linear.py" to produce the figures corresponding for 3 datasets. It is noted that each experiment is run at least 10 times and then the result is averaged.

- To produce the experiments for Linear Regresstion:
  ![linear_synthetic20train_loss](https://user-images.githubusercontent.com/44039773/86306668-3b85f480-bc58-11ea-8cae-b50e6f43eec0.png)
  - In folder data/linear_synthetic, before generating linear data set, configure the value of $\rho$ for example rho = 1.4 (in the papers we use 3 different values of $\rho$: 1.4, 2, 5) then run: "python3 generate_linear_regession.py" to generate data corresponding to different values of $\rho$.
  - To find the optimal solution: In folder data/linear_synthetic, run python3 optimal_solution_finding.py (also the value of $\rho$ need to be configured to find the optimal solution)
  - To generate result for the training process, run below commands:
    <pre><code>
    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 1.4 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 1.4 --times  10 
    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 1.4 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 1.4 --times  10 

    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 2 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 2 --times  10 
    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 2 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 2 --times  10 

    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 5 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 5 --times  10 
    python3 -u main.py --dataset logistic_synthetic --optimizer fedfedl --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 5 --times  10
    python3 -u main.py --dataset logistic_synthetic --optimizer fedsgd --model linear.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 5 --times  10 
    </code></pre>
  - All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results".
  - To produce the figure for linear regression run <pre><code> python3 main_linear.py</code></pre>
  
- For MNIST:
![mnist20test_accu](https://user-images.githubusercontent.com/44039773/86306670-3e80e500-bc58-11ea-8fec-5e80a3fcf08a.png)
![mnist20train_loss](https://user-images.githubusercontent.com/44039773/86306673-3f197b80-bc58-11ea-9efa-c7df0d88eaff.png)
    <pre><code>
    python3 -u main.py --dataset mnist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10
    python3 -u main.py --dataset mnist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 

    python3 -u main.py --dataset mnist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 40 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10
    python3 -u main.py --dataset mnist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 40 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 

    python3 -u main.py --dataset mnist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10
    python3 -u main.py --dataset mnist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 

    python3 -u main.py --dataset mnist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  2 --rho 0 --times  10
    python3 -u main.py --dataset mnist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  4 --rho 0 --times  10
    </code></pre>

- For FEMNIST:
![nist10test_accu](https://user-images.githubusercontent.com/44039773/86306675-3fb21200-bc58-11ea-9996-19c7f3898da5.png)
![nist10train_loss](https://user-images.githubusercontent.com/44039773/86306678-404aa880-bc58-11ea-97e7-fbfaa4df796e.png)
    <pre><code>
    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  10 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  10 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  10 --learning_rate  0.015 --hyper_learning_rate  0.5 --rho 0 --times  10 

    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  20 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  20 --learning_rate  0.015 --hyper_learning_rate  0.5 --rho 0 --times  10 

    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  40 --learning_rate  0.003 --hyper_learning_rate  0.2 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedsgd --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 20 --num_epochs  40 --learning_rate  0.003 --hyper_learning_rate  0 --rho 0 --times  10 
    python3 -u main.py --dataset nist --optimizer fedfedl --model mclr.py --num_rounds  800 --clients_per_round 10 --batch_size 0 --num_epochs  40 --learning_rate  0.015 --hyper_learning_rate  0.5 --rho 0 --times  10 
    </code></pre>

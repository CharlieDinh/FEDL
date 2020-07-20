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
- To generate FEMNIST Data: first In folder data/nist run preprocess.sh to obtain all raw data, or can be download in the link below, then run  python3 generate_niid_femnist_100users.py
- To generate niid Linear Synthetic: In folder data/linear_synthetic, run: "python3 generate_linear_regession.py" 
- The datasets are available to download at: https://drive.google.com/drive/folders/1Q91NCGcpHQjB3bXJTvtx5qZ-TrIZ9WzT?usp=sharing

# Produce figures in the paper:
- There is a main file "main.py" which allows running all experiments and 3 files "main_mnist.py, main_nist.py, main_linear.py" to produce the figures corresponding for 3 datasets. It is noted that each experiment is run at least 10 times and then the result is averaged.

- To produce the experiments for Linear Regresstion:
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/87938445-f546e700-cad9-11ea-8138-a0b6f5e68419.png" height="300">
</p>

  - In folder data/linear_synthetic, before generating linear data set, configure the value of $\rho$ for example rho = 1.4 (in the papers we use 3 different values of $\rho$: 1.4, 2, 5) then run: "python3 generate_linear_regession.py" to generate data corresponding to different values of $\rho$.
  - To find the optimal solution: In folder data/linear_synthetic, run python3 optimal_solution_finding.py (also the value of $\rho$ need to be configured to find the optimal solution)
  - To generate result for the training process, run below commands:
    <pre><code>
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 1.4 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 1.4 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 1.4 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 1.4 --times  1 

    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 2 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 2 --times  1 
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 2 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 2 --times  1 

    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.1 --rho 5 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.3 --rho 5 --times  1 
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.5 --rho 5 --times  1
    python3 -u main.py --dataset linear_synthetic --optimizer fedfedl --model linear.py --num_rounds  200 --clients_per_round 100 --batch_size 0 --num_epochs  20 --learning_rate  0.01 --hyper_learning_rate  0.7 --rho 5 --times  1 
    </code></pre>
  - All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results".
  - To produce the figure for linear regression run <pre><code> python3 main_linear.py</code></pre>
  - Note that all users are selected in Synthetic data, so the experiments for each case of synthetic only need to be run once
  
- For MNIST, run below commands:
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
    
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/87938456-f8da6e00-cad9-11ea-8ed2-9dbf8f5d245d.png" height="300">
  
  <img src="https://user-images.githubusercontent.com/44039773/87938464-fa0b9b00-cad9-11ea-9a5f-b68e52b4f13d.png" height="300">
</p>

  - To produce the figure for MNIST experiment, run <pre><code> python3 main_mn.py</code></pre>
  
- For FEMNIST, run below commands:
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
    
<p align="center">
  <img src="https://user-images.githubusercontent.com/44039773/87938469-fb3cc800-cad9-11ea-9af5-11f91ba14e5b.png" height="300">
  
  <img src="https://user-images.githubusercontent.com/44039773/87938476-fd068b80-cad9-11ea-95cb-72f075ab1471.png" height="300">
</p>

  - To produce the figure for FEMNIST experiment, run <pre><code> python3 main_nist.py</code></pre>

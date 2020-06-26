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
- There are 3 main files to produce the figures and results corresponding for 3 datasets.\
- For Linear Regresstion:
  - To find the optiomal solution: In folder data/linear_synthetic, run python3 optimal_solution_finding.py
- For MNIST:
- For FEMNIST:

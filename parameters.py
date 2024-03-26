import numpy as np
import torch


input_size = 2
hidden_size = 1
output_size = 1
batch_size = 5
lr=0.01
epochs= 10

num_trajectories = 10**3
num_intervals = 10
T = 10
x_init = 0.0
dim = 1
u = torch.tensor([0.0])
#if you have to progressively change a as well, then need to remove param.a from functions in main
a=0.2
eps_big=1
eps = 0.01
common_ratio = 0.7
num_in_sequence = -int(np.log(1/eps)/np.log(common_ratio))+1
single_sample_size = 10**5
seq_sample_size = np.full(num_in_sequence+1, single_sample_size)
eps_start = eps*common_ratio**(-num_in_sequence)
eps_seq = eps_start * np.power(common_ratio, np.arange(num_in_sequence+1))
initial_step_size = 0.01
step_size_seq = np.full(num_in_sequence+1, initial_step_size)

rule= 'simple_brownian'
theta_ou= 1
theta_cir=1
theta_dw=1
brownian_diff=1
ou_diff=1
cir_diff=1
dw_diff=1




num_iterations= 10**3
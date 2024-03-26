import numpy as np

#figure out how to include other sdes here
grad_potential_dict = {'simple_brownian': 0}


def generate_sde(rule, T, x_init, dim, num_trajectories, num_intervals, u, eps):
  sdes= []
  grad_potential= grad_potential_dict[rule]
  dt= T/(num_intervals)
  mean = np.zeros(dim)
  var = np.double(dt)
  if(dim>1):
    var= calculate_covariance(dim, var, T)
  for _ in range(num_trajectories):
    sde= [x_init]
    bm = np.random.normal(mean,var,int(num_intervals))
    for i in range(int(num_intervals)):
      current_value = sde[-1] - grad_potential * dt + np.sqrt(2) * (u.detach().numpy().item() * dt) + np.sqrt(2*eps) * bm[i]
      sde = np.concatenate((sde, [current_value]))
    sdes.append(sde)
  return sdes


def generate_time(num_trajectories, num_intervals, T):
  t=[]
  dt= T/num_intervals
  for _ in range(num_trajectories):
    t_=[T-(i)*dt for i in range(int(num_intervals+1))]
    t.append(t_)
  return t


def calculate_covariance(d, var, t):
    covariance_matrix = np.zeros((d, d), dtype=np.double)

    for i in range(d):
        for j in range(d):
            covariance_matrix[i, j] = min(i, j) * var * t

    return covariance_matrix


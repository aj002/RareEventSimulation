import numpy as np
import parameters as param

def grad_potential(x, rule):
   if(rule== 'simple_brownian'):
      return 0
   elif(rule== 'ornstein_uhlenbeck'):
      return param.theta_ou*x
   elif(rule=='cir'):
      return param.theta_cir*x
   elif(rule=='double_well'):
      return param.theta_dw*4*x(x**2-1)



def diffusion_term(x, rule):
   if(rule== 'simple_brownian'):
      return param.brownian_diff
   elif(rule== 'ornstein_uhlenbeck'):
      return param.ou_diff
   elif(rule=='cir'):
      return param.cir_diff*np.sqrt(x)
   elif(rule=="double_well"):
      return param.dw_diff


def generate_sde(rule, T, x_init, dim, num_trajectories, num_intervals, u, eps):
  sdes= []
  dt= T/(num_intervals)
  mean = np.zeros(dim)
  var = np.double(dt)
  if(dim>1):
    var= calculate_covariance(dim, var, T)
  for _ in range(num_trajectories):
    sde= [x_init]
    bm = np.random.normal(mean,var,int(num_intervals))
    for i in range(int(num_intervals)):
      current_value = sde[-1] - grad_potential(sde[-1], rule) * dt + np.sqrt(2) * (u.detach().numpy().item() * dt) + diffusion_term(sde[-1], rule)* np.sqrt(2*eps) * bm[i]
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


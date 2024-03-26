import parameters as param
import torch
from data_generator import*


def objective_fun(u, num_samples, eps, rule):
    values = []
    for _ in range(num_samples):
        sde = generate_sde(rule, param.T, param.x_init, param.dim, 1, param.num_intervals, u, eps)
        sde = np.concatenate(sde)
        if sde[-1] < param.a:
            values.append(0.0)
        else:
            sde_tensor = torch.tensor(sde)  # Convert numpy array to PyTorch tensor
            value = torch.exp(-u/np.sqrt(eps)*sde_tensor[-1]-u**2/(2*eps)*param.T)
            values.append(value.item())  # Append only the item if it's a single value
    return np.mean(values), np.var(values)




# def gradient_fun(u, T, a, eps, num_intervals):
#     mean = np.zeros(1)
#     var = np.double(T/num_intervals)
#     bm = np.random.normal(mean,var,num_intervals)
#     sde = [0]
#     dt = 1/num_intervals
#     weights = [1,0.5,1]
#     for i in range(num_intervals):
#         grad_potential = 0
#         sde = np.concatenate((sde,[sde[-1]+-grad_potential+np.sqrt(2)*u*dt+np.sqrt(2*eps)*bm[i]]))
#     if sde[-1] >= a:
#         return weights[0]*np.array([1.0,T,0.0])-weights[1]*np.array([0.0, T*(x[1]+x[2]**2/2), T*x[2]*(x[1]+x[2]**2/2)])-weights[2]*np.array([2*(x[0]+x[2]*sde[-1]), 0,2*sde[-1]*(x[0]+x[2]*sde[-1])])
#     else:
#         return weights[0]*np.array([1.0,T,0])-weights[1]*np.array([0, T*(x[1]+x[2]**2/2), T*x[2]*(x[1]+x[2]**2/2)])
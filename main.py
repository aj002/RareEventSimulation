import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
np.random.seed(42)
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import parameters as param
from data_generator import*
from helper_functions import*
import neural_net 
from optimization_functions import*
import csv
import theoretical_calcs as theory

print(param.num_in_sequence)


def sequential_algorithm(eps_seq):
    trajectory = []
    objective_values = []
    variances = []
    u= torch.tensor(0.0)
    model = neural_net.TwoLayerNet(input_size=param.input_size, hidden_size=param.hidden_size, output_size=param.output_size)
    for ind in range(len(eps_seq)):
        eps = eps_seq[ind]
        print(f"eps: {eps}")
        u, traj, obj, vs, model= neural_approximation(u, model, eps)
        trajectory.append(traj.copy())
        objective_values.append(obj)
        variances.append(vs)
    plt.plot(objective_values)
    return objective_values, variances



def neural_approximation(u, model, eps):
    X_tensor = torch.FloatTensor(generate_sde(param.rule, param.T, param.x_init, param.dim, param.num_trajectories, param.num_intervals, u, eps))
    t_tensor= torch.FloatTensor(generate_time(param.num_trajectories, param.num_intervals, param.T))
    X_tensor.requires_grad_(True)
    t_tensor.requires_grad_(True)
    # X_tensor.retain_grad()
    # t_tensor.retain_grad()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=param.lr)
    for _ in range(param.epochs):
        output, grads = compute_OutputAndGrads(X_tensor, t_tensor, model)
        # print("Output:", output)
        # print("Grads:", grads)
        weights = [1,2,1]
        grad_potential = grad_potential_dict[param.rule]
        time_points = np.linspace(0, param.T, param.num_intervals+1)
        
        initial_condition= torch.mean(output[:, 0])

        hjb_tensor = torch.mean((grads[:, :, 1] - grad_potential * grads[:, :, 0] - ((grads[:, :, 0]) ** 2) / 2) ** 2, dim=0).detach().numpy()
        hjb_tensor = hjb_tensor.squeeze()
        integral_hjb= np.trapz(hjb_tensor, time_points)

        last_time_step_values = output[:, -1]
        terminal_condition = torch.where(last_time_step_values > param.a, (last_time_step_values-torch.tensor(1))**2, (last_time_step_values-torch.tensor(0))**2)
        mean_terminal_condition = torch.mean(terminal_condition).item()

        initial_condition_values.append(initial_condition.item())
        integral_hjb_values.append(integral_hjb)
        mean_terminal_condition_values.append(mean_terminal_condition)
        #variance_array.append()
        current_weights = [param.data.numpy() for _, param in model.named_parameters()]
        weights_values.append(current_weights)

        loss_function = (-weights[0] * initial_condition + weights[1] * integral_hjb + weights[2] * mean_terminal_condition)
        loss_array.append(loss_function.item())
        threshold.append(param.a/np.sqrt(eps))
        target_tensor = torch.tensor(0.0)
        loss = criterion(loss_function, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (epoch + 1) % 100 == 0:
         #   print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #final_solution= model.evaluate(X_tensor, t_tensor)
    outputs, grads= compute_OutputAndGrads(X_tensor, t_tensor, model)
    outputs_np = outputs.detach().numpy()

    # Flatten the 3D array
    flattened_outputs = outputs_np.reshape(outputs_np.shape[0], -1)

    # Specify the file path where you want to save the CSV file
    csv_file_path = 'outputs.csv'

    # Save the flattened 2D array as a CSV file
    np.savetxt(csv_file_path, flattened_outputs, delimiter=',')

    print(f"Outputs saved to {csv_file_path}")





    u = -torch.mean(grads[:, :, 0]) / torch.sqrt(torch.tensor(2.0))
    traj= [param.data for _, param in model.named_parameters()]
    rule= 'simple_brownian'
    obj, vs= objective_fun(u, param.num_iterations, eps, rule)
    theoretical_prob= theory.exceedance_probability_Brownian_motion(param.a/np.sqrt(2*eps),10)
    print("u:", u)
    print("obj:", obj)
    print("theo:", theoretical_prob)
    print("traj:", traj)
    return u,traj, obj, vs, model



def compute_OutputAndGrads(X_tensor, t_tensor, model):
    all_outputs = []
    all_grads = []

    for t_idx in range(t_tensor.shape[1]):
        outputs_t = []
        grads_t = []
        for x_idx in range(X_tensor.shape[0]):
            t_tensor = torch.tensor([[t_idx]], dtype=torch.float32, requires_grad=True)
            x_tensor = torch.tensor([[X_tensor[x_idx][t_idx]]], dtype=torch.float32, requires_grad=True)
            output = model(x_tensor, t_tensor)
            output.backward(create_graph=True)
            grad_x = x_tensor.grad
            grad_t = t_tensor.grad
            grads = [grad_x, grad_t]
            outputs_t.append(output)
            grads_t.append(grads)

        all_outputs.append(torch.cat(outputs_t, dim=0))

        # Convert the list of gradients to a tensor before concatenation
        grads_tensor = torch.stack([torch.cat(grads_timestep, dim=0) for grads_timestep in zip(*grads_t)], dim=1)
        all_grads.append(grads_tensor)

    all_outputs_tensor = torch.stack(all_outputs, dim=1)
    all_grads_tensor = torch.stack(all_grads, dim=1)
    return all_outputs_tensor, all_grads_tensor


csv_file_path = 'output.csv'

initial_condition_values=[]
integral_hjb_values = []
mean_terminal_condition_values = []
loss_array=[]
threshold=[]
weights_values = []
variance_array=[]

objective_function, var= objective_fun(param.u, param.num_iterations, param.eps_big, param.rule)
sequential_algorithm(param.eps_seq)
rows = zip(initial_condition_values, integral_hjb_values, mean_terminal_condition_values, loss_array, variance_array, threshold)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Initial Condition', 'Integral HJB', 'Mean Terminal Condition', 'Loss', 'Variance', 'Threshold'])
    writer.writerows(rows)

weights_csv_file_path = 'weights.csv'
with open(weights_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Layer', 'Parameter', 'Weight'])
    for i, weights in enumerate(weights_values):
        for j, w in enumerate(weights):
            writer.writerow([f'Layer {i}', f'Parameter {j}', w])





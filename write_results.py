import csv

def write_loss_functions(file_path, initial_condition_values, integral_hjb_values, mean_terminal_condition_values, loss_array, variance_array, threshold):
    rows = zip(initial_condition_values, integral_hjb_values, mean_terminal_condition_values, loss_array, variance_array, threshold)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Initial Condition', 'Integral HJB', 'Mean Terminal Condition', 'Loss', 'Variance', 'Threshold'])
        writer.writerows(rows)


def write_weights(file_path, weights_values):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Layer', 'Parameter', 'Weight'])
        for i, weights in enumerate(weights_values):
            for j, w in enumerate(weights):
                writer.writerow([f'Layer {i}', f'Parameter {j}', w])
import numpy as np
import matplotlib.pyplot as plt

# Question 1 Part A
# Initializing random values for weights
weight_0 = np.random.uniform(-1/4, 1/4)  # weight_0 between [-1/4, 1/4]
weight_1 = np.random.uniform(-1, 1)      # weight_1 between [-1, 1]
weight_2 = np.random.uniform(-1, 1)      # weight_2 between [-1, 1]

# Combining the weights into a vector
true_weights = np.array([weight_0, weight_1, weight_2])
print(f"The randomly generated weights are {true_weights}")

# Question 1 Part B
# Creating sample data and corresponding labels
samples_count = 100
x_coord_1 = np.random.uniform(-1, 1, samples_count)
x_coord_2 = np.random.uniform(-1, 1, samples_count)

# Constructing the data matrix with a bias term
data_matrix = np.column_stack((np.ones(samples_count), x_coord_1, x_coord_2))

# Using the step function to generate labels
labels = (np.dot(data_matrix, true_weights) >= 0).astype(int)

# Plotting the generated data points
plt.figure(figsize=(8, 8))
plt.scatter(x_coord_1[labels == 1], x_coord_2[labels == 1], color='red', label='Class 1 (Label=1)')
plt.scatter(x_coord_1[labels == 0], x_coord_2[labels == 0], color='blue', label='Class 0 (Label=0)')

# Plotting the decision boundary line: weight_0 + weight_1*x1 + weight_2*x2 = 0
x_boundary_vals = np.linspace(-1, 1, 100)
y_boundary_vals = -(true_weights[1] * x_boundary_vals + true_weights[0]) / true_weights[2]
plt.plot(x_boundary_vals, y_boundary_vals, label='Decision Boundary', color='black')

plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Generated Data and Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# Question 1 Part C
# Calculating the normal vector and distance from the origin
normal_vector = np.array([weight_1, weight_2])
normal_magnitude = np.linalg.norm(normal_vector)

# Distance from the origin to the line
distance_to_origin = np.abs(weight_0) / normal_magnitude

print(f"Normal vector: {normal_vector}")
print(f"Magnitude of the normal vector: {normal_magnitude}")
print(f"Distance from the origin to the decision boundary: {distance_to_origin}")

# Plotting the line, normal vector, and the shortest distance to origin
def plot_line_and_normal(weight_0, weight_1, weight_2):
    x_vals = np.linspace(-1, 1, 100)
    y_vals = (-weight_0 - weight_1 * x_vals) / weight_2

    plt.plot(x_vals, y_vals, label=f'Line: {weight_0:.2f} + {weight_1:.2f}*x1 + {weight_2:.2f}*x2 = 0')
    plt.quiver(0, 0, weight_1, weight_2, angles='xy', scale_units='xy', scale=1, color='r', label='Normal Vector')

    distance = abs(weight_0) / np.sqrt(weight_1**2 + weight_2**2)
    closest_x1 = -weight_1 * weight_0 / (weight_1**2 + weight_2**2)
    closest_x2 = -weight_2 * weight_0 / (weight_1**2 + weight_2**2)

    plt.plot([0, closest_x1], [0, closest_x2], 'g--', label=f'Distance = {distance:.4f}')
    plt.scatter(closest_x1, closest_x2, color='green', zorder=5)
    plt.scatter(0, 0, color='black', zorder=5)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("Line, Normal Vector, and Distance from Origin")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Function to compute the distance from the origin to the line
def calculate_distance_from_origin(weight_0, weight_1, weight_2):
    return abs(weight_0) / np.sqrt(weight_1**2 + weight_2**2)

# Plot and calculate the distance
plot_line_and_normal(weight_0, weight_1, weight_2)
distance = calculate_distance_from_origin(weight_0, weight_1, weight_2)
print(f"The distance from the origin to the decision boundary is: {distance:.4f}")

# Question 2 Part A
# Function to retrieve generated weights
def get_weights():
    return np.array([weight_0, weight_1, weight_2])

# Function to generate random data
def create_sample_data(samples=100):
    sample_points = np.random.uniform(-1, 1, (samples, 2))
    sample_points = np.hstack((np.ones((samples, 1)), sample_points))
    return sample_points

# Function to generate labels based on true weights
def create_labels(data, true_weights):
    return np.sign(data @ true_weights)

# Perceptron algorithm with custom update rule
def custom_perceptron_algorithm(data, labels, learning_rate=1):
    weights = np.array([1.0, 1.0, 1.0])
    num_samples = data.shape[0]
    error_list = []
    epoch = 0
    is_converged = False

    while not is_converged:
        total_errors = 0
        for i in range(num_samples):
            prediction = np.sign(np.dot(weights, data[i]))
            if labels[i] != prediction:
                weights += learning_rate * data[i] * (labels[i] - prediction)
                total_errors += 1
        error_list.append(total_errors)

        if total_errors == 0:
            is_converged = True

        epoch += 1

    return weights, error_list

# Function to plot decision boundary
def visualize_decision_boundary(weights, data, labels, true_weights):
    x_vals = np.linspace(-1, 1, 100)
    decision_boundary = (-weights[0] - weights[1] * x_vals) / weights[2]
    true_boundary = (-true_weights[0] - true_weights[1] * x_vals) / true_weights[2]

    plt.scatter(data[labels == 1, 1], data[labels == 1, 2], color='red', label='Class 1')
    plt.scatter(data[labels == -1, 1], data[labels == -1, 2], color='blue', label='Class 0')

    plt.plot(x_vals, decision_boundary, label='Learned Boundary', color='green')
    plt.plot(x_vals, true_boundary, label='True Boundary', linestyle='--', color='black')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("Decision Boundary by Perceptron")
    plt.grid(True)
    plt.show()

# Main execution for Part 2
np.random.seed(0)

# Step 1: Generate data and labels
true_weights = get_weights()
data = create_sample_data(samples=100)
labels = create_labels(data, true_weights)

# Step 2: Apply Perceptron algorithm
learned_weights, errors = custom_perceptron_algorithm(data, labels, learning_rate=1)

# Step 3: Output results and plot the decision boundary
print(f"Learned weights: {learned_weights}")
print(f"True weights: {true_weights}")
print(f"Epochs until convergence: {len(errors)}")
visualize_decision_boundary(learned_weights, data, labels, true_weights)

# Plot errors per epoch
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('Error Reduction per Epoch')
plt.grid(True)
plt.show()

# Question 2 Part B
# Run the Perceptron algorithm for multiple learning rates and plot results
learning_rates = [0.1, 1, 10]
for rate in learning_rates:
    learned_weights, errors = custom_perceptron_algorithm(data, labels, learning_rate=rate)
    print(f"Learned weights for learning rate = {rate}: {learned_weights}")
    print(f"Epochs to convergence: {len(errors)}")

    plt.plot(errors, label=f'Learning rate = {rate}')

plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.legend()
plt.title('Error per Epoch for Different Learning Rates')
plt.grid(True)
plt.show()

# Question 2 Part C
# Function to retrieve previously generated weights
def get_true_weights():
    return np.array([weight_0, weight_1, weight_2])

# Generate 1000 samples of (x1, x2)
def generate_large_dataset(sample_size=1000):
    sample_points = np.random.uniform(-1, 1, (sample_size, 2))  # Generate points in [-1, 1]^2
    sample_points = np.hstack((np.ones((sample_size, 1)), sample_points))  # Add bias term x0 = 1
    return sample_points

# Create labels based on the true weights
def create_labels_for_large_data(data, true_weights):
    return np.sign(data @ true_weights)

# Custom Perceptron algorithm with weight update rule
def perceptron_with_adjusted_weights(data, labels, learning_rate=1):
    # Initialize weights to [1.0, 1.0, 1.0] as floats
    learned_weights = np.array([1.0, 1.0, 1.0])
    num_samples = data.shape[0]
    
    error_tracking = []  # Keep track of the number of errors per epoch
    epochs = 0
    has_converged = False

    while not has_converged:  # Loop until no misclassifications
        total_errors = 0
        for i in range(num_samples):
            prediction = np.sign(np.dot(learned_weights, data[i]))
            if labels[i] != prediction:  # Misclassification
                error_point = data[i]  # The misclassified point
                # Update the weights
                learned_weights += learning_rate * error_point * (labels[i] - prediction)
                total_errors += 1
        error_tracking.append(total_errors)

        if total_errors == 0:
            has_converged = True

        epochs += 1

    return learned_weights, error_tracking

# Function to plot the decision boundary learned by the perceptron
def visualize_perceptron_boundary(learned_weights, data, labels, true_weights):
    x_range_vals = np.linspace(-1, 1, 100)
    # Decision boundary using the learned weights
    decision_boundary_vals = (-learned_weights[0] - learned_weights[1] * x_range_vals) / learned_weights[2]
    # Decision boundary using the true weights
    true_boundary_vals = (-true_weights[0] - true_weights[1] * x_range_vals) / true_weights[2]

    # Plot data points
    plt.scatter(data[labels == 1, 1], data[labels == 1, 2], color='red', label='Class 1')
    plt.scatter(data[labels == -1, 1], data[labels == -1, 2], color='blue', label='Class 0')

    # Plot the decision boundaries
    plt.plot(x_range_vals, decision_boundary_vals, label='Learned Boundary', color='green')
    plt.plot(x_range_vals, true_boundary_vals, label='True Boundary', linestyle='--', color='black')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.grid(True)
    plt.show()

# Main execution for Question 2 Part C
np.random.seed(0)

# Step 1: Generate large dataset and labels
true_weights = get_true_weights()  # Retrieve the true weights
large_data = generate_large_dataset(sample_size=1000)  # Generate 1000 data samples
labels_for_large_data = create_labels_for_large_data(large_data, true_weights)  # Generate labels

# Step 2: Run Perceptron algorithm with custom update rule
learned_weights_large_data, errors_large_data = perceptron_with_adjusted_weights(large_data, labels_for_large_data, learning_rate=1)

# Step 3: Display results and plot the decision boundary
print(f"Learned weights: {learned_weights_large_data}")
print(f"True weights: {true_weights}")
print(f"Number of epochs to convergence: {len(errors_large_data)}")

# Step 4: Plot the decision boundary
visualize_perceptron_boundary(learned_weights_large_data, large_data, labels_for_large_data, true_weights)

# Plot the errors per epoch
plt.plot(errors_large_data)
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('Errors per Epoch (Perceptron Convergence)')
plt.grid(True)
plt.show()

# Question 2 Part D
# Perceptron with a maximum number of epochs
def perceptron_algorithm(data, labels, learning_rate, max_epochs=100):
    weights = np.random.uniform(-1, 1, data.shape[1])  # Initialize weights randomly
    epoch_errors = []

    for epoch in range(max_epochs):
        errors = 0
        for i in range(len(data)):
            if labels[i] != np.sign(np.dot(data[i], weights)):
                weights += learning_rate * labels[i] * data[i]  # Update weights
                errors += 1
        epoch_errors.append(errors)
        if errors == 0:
            break

    # Ensure epoch_errors has max_epochs length
    if len(epoch_errors) < max_epochs:
        epoch_errors.extend([epoch_errors[-1]] * (max_epochs - len(epoch_errors)))
    return weights, epoch_errors

# Run multiple experiments and compute error statistics
def perform_experiments(data, labels, learning_rate, repetitions=100):
    all_error_rates = []
    for _ in range(repetitions):
        _, errors = perceptron_algorithm(data, labels, learning_rate)
        all_error_rates.append(errors)
    return np.array(all_error_rates)

# Run the experiment for different learning rates
learning_rates = [0.1, 1, 10]
for rate in learning_rates:
    experiment_errors = perform_experiments(large_data, labels_for_large_data, rate)

    # Calculate average and percentiles
    avg_errors = np.mean(experiment_errors, axis=0)
    percentile_10 = np.percentile(experiment_errors, 10, axis=0)
    percentile_90 = np.percentile(experiment_errors, 90, axis=0)

    # Plot the results
    plt.plot(avg_errors, label=f'Learning Rate = {rate}')
    plt.fill_between(range(len(avg_errors)), percentile_10, percentile_90, alpha=0.2)

plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.legend()
plt.title('Average Errors per Epoch for Different Learning Rates')
plt.grid(True)
plt.show()


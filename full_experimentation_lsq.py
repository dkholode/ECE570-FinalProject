# The code below is for the Gaussian kernel with fixed bandwidth, k(x,y) := exp(-||x-y||_2^2).
# The bandwidth can be changed by scaling the input point coordinates.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


### Auxiliary : computing squared-distance matrix ###

def get_sqdistance_matrix(M1, M2):
    allsqnorms = np.linalg.norm(np.vstack([M1,M2]), axis=1).reshape(-1, 1)**2
    M1sqnorms = allsqnorms[:M1.shape[0],:]
    M2sqnorms = allsqnorms[M1.shape[0]:,:].reshape(1, -1)
    dm = M1sqnorms + M2sqnorms - 2.0 * np.dot(M1, M2.T)
    dm[dm < 0.0] = 0.0
    return dm


### Exact KDE ###

def GaussianKDE(dataset, queries):
    exp_sq_dist_matrix = np.exp(-1 * get_sqdistance_matrix(dataset, queries))
    return np.mean(exp_sq_dist_matrix, axis=0).T


### LSQ with Random Fourier Features ###

class LSQ_RFF:

    def __init__(self, dataset, dimension, reps):
        self.n = None
        self.d = dimension
        self.reps = reps

        # Sample random fourier features
        self.rff = np.sqrt(2) * np.random.normal(0, 1, (self.d, self.reps))
        self.rff_shift = np.random.uniform(0, 2*np.pi, self.reps).reshape(1, -1)

        self.rff_kde = None
        self.sanitized_rff_kde = None

        self.sketch_dataset(dataset)

    def sketch_dataset(self, dataset):
        self.n = dataset.shape[0]
        self.rff_kde = np.mean(self.apply_rff(dataset), axis=0)

    def apply_rff(self, m):
        return np.sqrt(2) * np.cos(np.dot(m, self.rff) + self.rff_shift)

    def sanitize(self, epsilon):
        self.sanitized_rff_kde = self.rff_kde + \
                                 np.random.laplace(0, np.sqrt(2) * self.reps * 1. / (epsilon * self.n), self.reps)

    def non_private_kde(self, queries):
        return (1./self.reps) * np.dot(self.rff_kde, self.apply_rff(queries).T)

    def private_kde(self, queries):
        return (1./self.reps) * np.dot(self.sanitized_rff_kde, self.apply_rff(queries).T)


### LSQ with Fast Gauss Transform ###

class LSQ_FGT:

    def __init__(self, dataset, dimension, coordinate_range, rho):
        self.n = None
        self.d = dimension
        self.coordinate_range = coordinate_range
        self.small_radius_squared = rho
        self.rho = rho

        # Sketch
        self.sketch = np.zeros((self.coordinate_range ** self.d, self.rho ** self.d))
        self.sanitized_sketch = None

        # Sketch indexing auxiliaries
        self.aux_dim0_powers = np.flip(np.array([self.coordinate_range ** i for i in range(self.d)]))
        self.aux_dim0_tuples = np.indices(tuple([self.coordinate_range] * self.d)).reshape(self.d, -1).T
        self.aux_dim1_tuples = np.indices(tuple([self.rho] * self.d)).reshape(self.d, -1).T
        self.noise_scale = (2 * (1 - 0.5 ** self.rho)) ** self.d

        # Hermite polynomials
        self.hermite_polynomials = [sp.special.hermite(j) for j in np.arange(self.rho)]

        self.sketch_dataset(dataset)

    def sketch_dataset(self, dataset):
        self.n = dataset.shape[0]

        if self.rho == 0:
            return

        # Partition dataset into hypercubes
        rounded_dataset = np.rint(dataset)
        # Compute the index of the cell containing each data point
        cell_indices = rounded_dataset.dot(self.aux_dim0_powers).astype(np.dtype(int))

        if self.rho == 1:
            np.add.at(self.sketch[:, 0], cell_indices, np.ones(self.n))
        elif self.d == 2:
            residual_dataset = dataset - rounded_dataset
            all_powers = np.einsum('nk,nl->nkl',
                                   np.vstack([residual_dataset[:, 0]**j for j in range(self.rho)]).T,
                                   np.vstack([residual_dataset[:, 1]**j for j in range(self.rho)]).T
                                   ).reshape(self.n, -1)
            np.add.at(self.sketch, cell_indices, all_powers)
        else:
            residual_dataset = dataset - rounded_dataset
            for idx, idx_tuple in enumerate(self.aux_dim1_tuples):
                np.add.at(self.sketch[:, idx], cell_indices, np.prod(np.power(residual_dataset, idx_tuple), axis=1))

        self.sketch *= 1. / self.n

    def sanitize(self, epsilon):
        self.sanitized_sketch = self.sketch + \
                                np.random.laplace(0, self.noise_scale * 1. / (epsilon * self.n), self.sketch.shape)

    def g(self, query):

        # Compute the cells which are close enough to matter
        cell_distances = get_sqdistance_matrix(query.reshape(1, -1), self.aux_dim0_tuples)
        relevant_cells = np.where(cell_distances.ravel() <= self.small_radius_squared)[0]

        # Compute normalized hermite functions of all query residual coordinates in relevant cells
        # (Denominator turns hermite polynomial to hermite function)
        q_residuals = query - self.aux_dim0_tuples[relevant_cells, :]
        denominator = np.exp(q_residuals ** 2)
        hermite_evaluations = [(1. / np.math.factorial(j)) *
                               self.hermite_polynomials[j](q_residuals) / denominator for j in np.arange(self.rho)]

        # Compute g-coordinates of query
        q_sketch = np.zeros((len(relevant_cells), self.rho ** self.d))
        for cell_id in range(len(relevant_cells)):
            for idx, idx_tuple in enumerate(self.aux_dim1_tuples):
                q_sketch[cell_id, idx] = np.prod([hermite_evaluations[idx_tuple[i]][cell_id, i] for i in range(self.d)])

        return q_sketch, relevant_cells

    def one_query_kde(self, query, sanitized):
        if sanitized:
            dataset_sketch = self.sanitized_sketch
        else:
            dataset_sketch = self.sketch
        q_sketch, relevant_cells = self.g(query)
        return dataset_sketch[relevant_cells, :].ravel().dot(q_sketch.ravel())

    def non_private_kde(self, queries):
        return np.array([self.one_query_kde(query, False) for query in queries])

    def private_kde(self, queries):
        return np.array([self.one_query_kde(query, True) for query in queries])


if __name__ == '__main__':
    ### Usage example ###

    # Generate random dataset and queries:
    dimension = 2
    coordinate_range = 10
    n_data = 100000
    n_queries = 20
    dataset = np.random.uniform(0, coordinate_range-1, (n_data, dimension))
    queries = np.random.uniform(0, coordinate_range-1, (n_queries, dimension))

    method = "lsq-rff" # or "lsq-fgt"
    print("DP-KDE method:", method)

    if method == "lsq-rff":
        # Init LSQ-RFF:
        num_features = 200
        mechanism = LSQ_RFF(dataset, dimension, num_features)
    elif method == "lsq-fgt":
        # init LSQ-FGT:
        rho = 4
        mechanism = LSQ_FGT(dataset, dimension, coordinate_range, rho)

    # Exact Gaussian KDE:
    exact_kde = GaussianKDE(dataset, queries)
    print("Exact:", exact_kde)
    # Non-DP KDE estimates:
    non_dp_kde_estimate = mechanism.non_private_kde(queries)
    print("Non-DP estimate:", non_dp_kde_estimate)
    print("Mean error:", np.mean(np.abs(exact_kde - non_dp_kde_estimate)))
    # DP KDE estimates:
    epsilon = 0.5
    mechanism.sanitize(epsilon)
    dp_kde_estimate = mechanism.private_kde(queries)
    print("DP estimate:", dp_kde_estimate)
    print("Mean error:", np.mean(np.abs(exact_kde - dp_kde_estimate)))

    ### Reimplemented Functions ###
    ### Improvement 1: Optimized Squared-Distance Matrix Calculation
    def optimized_get_sqdistance_matrix(M1, M2):
        M1_norm = np.sum(M1 ** 2, axis=1).reshape(-1, 1)
        M2_norm = np.sum(M2 ** 2, axis=1).reshape(1, -1)
        dm = M1_norm + M2_norm - 2 * np.dot(M1, M2.T)
        dm[dm < 0] = 0  # Correcting for any negative values due to floating point errors
        return dm

    ### Improvement 2: Enhanced Random Fourier Features Sampling
    def enhanced_rff_sampling(dimension, reps):
        rff = np.sqrt(2) * np.random.normal(0, 1, (dimension, reps))
        rff_shift = np.random.uniform(0, 2*np.pi, reps).reshape(1, -1)
        return rff, rff_shift

    ### Improvement 3: Adjusted Bandwidth for Gaussian KDE
    def adjusted_gaussian_kde(dataset, queries, bandwidth=1.0):
        exp_sq_dist_matrix = np.exp(-1 * optimized_get_sqdistance_matrix(dataset, queries) / bandwidth)
        return np.mean(exp_sq_dist_matrix, axis=0).T

    def test_bandwidth_range(dataset, queries, exact_kde, bandwidth_values):
        best_bandwidth = None
        lowest_error = float('inf')
        bandwidth_errors = []

        for bandwidth in bandwidth_values:
            # Use the adjusted Gaussian KDE function with the current bandwidth
            estimated_kde = adjusted_gaussian_kde(dataset, queries, bandwidth)
            
            # Calculate the mean error for the estimated KDE against the exact KDE
            mean_error = np.mean(np.abs(exact_kde - estimated_kde))
            
            # Record the error and check if this is the lowest error so far
            bandwidth_errors.append((bandwidth, mean_error))
            if mean_error < lowest_error:
                lowest_error = mean_error
                best_bandwidth = bandwidth

        # Return the best bandwidth and the list of errors for each bandwidth tested
        return best_bandwidth, bandwidth_errors

    ### Experimentation with subtraction value ###
    def experiment_with_subtraction(dataset, queries, subtraction_values):
        results = []
        for value in subtraction_values:
            optimized_kde = adjusted_gaussian_kde(dataset, queries, bandwidth=1.5)  # Example bandwidth adjustment
            adjusted_kde = optimized_kde - value
            mean_error = np.mean(np.abs(exact_kde - adjusted_kde))
            results.append((value, mean_error))
        return results
    
    ### Plotting and Testing ###
    #Bandwidth
    # Define a range of bandwidth values to try
    bandwidth_values = np.linspace(0.1, 2.0, 20) 

    # Assuming 'dataset', 'queries', and 'exact_kde' are defined earlier in your script
    best_bandwidth, bandwidth_errors = test_bandwidth_range(dataset, queries, exact_kde, bandwidth_values)

    # Output the best bandwidth and corresponding error
    print(f"Best Bandwidth: {best_bandwidth}")

    # Adding the bandwidth testing results to the visualization
    plt.figure(figsize=(10, 6))
    plt.plot([bw for bw, err in bandwidth_errors], [err for bw, err in bandwidth_errors], marker='o', label='Bandwidth Test')
    plt.axvline(x=best_bandwidth, color='r', linestyle='--', label=f'Best Bandwidth = {best_bandwidth:.2f}')
    plt.xlabel('Bandwidth')
    plt.ylabel('Mean Error')
    plt.title('Bandwidth Testing for KDE')
    plt.legend()
    plt.show()


    #Subtraction Experiment
    # Define a range of values to try for subtraction
    subtraction_values = np.linspace(-0.1, 1, 20)   # Example range from 0 to 1

    # Run the experimentation
    experiment_results = experiment_with_subtraction(dataset, queries, subtraction_values)

    # Find the value with the lowest mean error
    best_value, best_error = min(experiment_results, key=lambda x: x[1])
    print("Best subtraction value:", best_value)
    print("Best mean error:", best_error)

    # Visualization of the experimentation results
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in experiment_results], [x[1] for x in experiment_results], marker='o')
    plt.xlabel('Subtraction Value')
    plt.ylabel('Mean Error')
    plt.title('Experimentation with Subtraction Values')
    plt.show()

    #Visualization of all estimates
    best_adjusted_kde = adjusted_gaussian_kde(dataset, queries, bandwidth=1.5) - best_value
    plt.figure(figsize=(12, 6))
    plt.plot(exact_kde, label='Exact KDE', linewidth=2)
    plt.plot(non_dp_kde_estimate, label='Original Non-DP Estimate', linestyle='--')
    plt.plot(dp_kde_estimate, label='Original DP Estimate', linestyle='-.')
    plt.plot(best_adjusted_kde, label='Best Adjusted KDE', linestyle=':')
    plt.title('Comparison of KDE Estimates')
    plt.xlabel('Query Points')
    plt.ylabel('Density Estimate')
    plt.legend()
    plt.show()
# Imports
import numpy as np
from tqdm import tqdm


def pokie(truth, posterior, num_runs=100):
    """
    truth: array of shape (num_truth, q)
    posterior: array of shape (num_models, num_truth, num_posterior_samples, q)
    num_runs: number of Monte Carlo runs

    Returns
    -------
    average_total_probabilities: array of shape (num_models,)
        The average probabilities across runs for each model
    """
    # Validate shapes
    num_truth, dim = truth.shape
    num_models, truth_check, num_posterior_samples, dim_check = posterior.shape

    if num_truth != truth_check:
        raise ValueError("Number of truth samples doesn't match the second dimension of posterior.")
    if dim != dim_check:
        raise ValueError("Parameter dimension of truth doesn't match that of posterior.")

    total_probability = []
    total_calibration_probability = []
    n_over_N_values = []  # Store all n/N values

    for _ in tqdm(range(num_runs)):
        # Generate random centers: (num_truth, dim)
        centers = np.random.uniform(low=0, high=1, size=(num_truth, dim))

        # Probability array to fill: (num_models, num_truth)
        probability = np.zeros((num_models, num_truth))

        calibration_probability = np.zeros((num_models, num_truth))

        # Loop over each truth sample i
        for i in range(num_truth):
            curr_truth = truth[i]
            curr_center = centers[i]

            # Precompute squared distance between center & truth for k-check
            theta_gt_dist_sq = np.sqrt(np.sum((curr_center - curr_truth) ** 2))

            # Loop over each model j
            for j in range(num_models):
                curr_posterior = posterior[j, i]  # shape: (num_posterior_samples, dim)

                # Pick one random sample index
                idx = np.random.randint(low=0, high=len(curr_posterior))
                random_sample = curr_posterior[idx]

                # Create a mask to exclude the chosen sample
                mask = np.ones(len(curr_posterior), dtype=bool)
                mask[idx] = False

                # Compute squared distance between center & random sample
                theta_dist_sq = np.sqrt(np.sum((curr_center - random_sample) ** 2)) + 1e-4

                # Compute squared distance to *all* remaining posterior points
                dist_sq_to_posterior = np.sqrt(np.sum((curr_center - curr_posterior[mask]) ** 2, axis=1)) # shape: (num_posterior_samples - 1,)

                # Determine k by comparing the truth distance vs. random sample distance
                if theta_gt_dist_sq <= theta_dist_sq:
                    curr_k = 1
                else:
                    curr_k = 0

                # Count how many posterior samples lie within the radius set by random_sample
                n = np.sum(dist_sq_to_posterior < theta_dist_sq)
                N = len(curr_posterior) - 1

                n_over_N_values.append(n / N)  # Collect normalized count

                if n > N:
                    raise ValueError("n > N: unexpected counting error.")

                # Compute probability
                if curr_k == 1:
                    curr_prob = (n + 1) / (N + 2)
                else:
                    curr_prob = (N - n + 1) / (N + 2)

                # Min Possible Value: 1 / (N + 2) # Max Possible Value: (N + 1) / (N + 2)
                min_possible_value = 1 / (N + 2)
                max_possible_value = (N + 1) / (N + 2)

                # Make sure curr_prob is within the range [min_possible_value, max_possible_value]
                if not (min_possible_value <= curr_prob <= max_possible_value):
                    raise ValueError("curr_prob is out of bounds.")

                probability[j, i] = curr_prob 

                calibration_probability[j, i] = curr_prob / max_possible_value

        model_expectations = np.empty((len(posterior), 1))
        # probability is a (num_models, num_truth) --> Average across ground truths (axis=1)
        model_expectations = np.mean(probability, axis=1) # shape: (num_models,)
        # Store in total_probability
        total_probability.append(model_expectations)

        model_calibraton_expectations = np.empty((len(posterior), 1))
        # probability is a (num_models, num_truth) --> Average across ground truths (axis=1)
        model_calibraton_expectations = np.mean(calibration_probability, axis=1)
        # Store in total_calibration_probability
        total_calibration_probability.append(model_calibraton_expectations)


    # Convert to numpy array
    total_probability = np.array(total_probability) # shape: (num_runs, num_models)
    # Average across runs (axis=0)
    average_total_probabilities = np.mean(total_probability, axis=0) # shape: (num_models,)

    # Convert to numpy array
    total_calibration_probability = np.array(total_calibration_probability) # shape: (num_runs, num_models)
    # Average across runs (axis=0)
    average_calibration_probabilities = np.mean(total_calibration_probability, axis=0) # shape: (num_models,)

    # Check to see if more than one row in total_probability
    if total_probability.shape[1] > 1:
        # print(f'Shape of total_probability: {total_probability.shape}')
        average_total_probabilities = average_total_probabilities / np.sum(average_total_probabilities)

    return average_total_probabilities, average_calibration_probabilities, np.array(n_over_N_values)

'''
Figure out how to handle multiple posterior models and still do bootstrapping (look at Pokie and it handles it)
'''
def pokie_bootstrap(truth, posterior, num_bootstrap: int = 100,):
    '''
    This will bootstrap & send it to Pokie which will return the results and store it to then return after all runs are done.
    '''

    store_average_probabilities = []
    store_calibration_probabilities = []
    store_n_over_N_values = []

    for i in range(num_bootstrap):
        # Bootstrap the truth
        idx = np.random.randint(low=0, high=len(truth), size=len(truth))
        
        # Sample with replacement from the full set of simulations
        boot_samples = posterior[:, idx, :, :]
        boot_theta = truth[idx]

        # Send to Pokie
        average_total_probabilities, average_calibration_probabilities, n_over_N = pokie(boot_theta, boot_samples)

        # Store the results
        store_average_probabilities.append(average_total_probabilities)
        store_calibration_probabilities.append(average_calibration_probabilities)
        store_n_over_N_values.append(n_over_N)

    # Convert to numpy array
    store_average_probabilities = np.array(store_average_probabilities) # shape: (num_runs, num_models)
   
    # Convert to numpy array
    store_calibration_probabilities = np.array(store_calibration_probabilities) # shape: (num_runs, num_models)
    store_n_over_N_values = np.concatenate(store_n_over_N_values)

    return store_average_probabilities, store_calibration_probabilities, store_n_over_N_values
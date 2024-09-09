import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_center(truth, dim):
    """
    Generate centers for each truth sample, with each center associated with a value of k (0 or 1).

    Parameters:
    truth (ndarray): Ground truth values, a 2D array of shape (num_ground_truth, dim).
    dim (int): Dimension of the ground truth values.

    Returns:
    ndarray: Centers of the hyper-sphere with shape (num_ground_truth), where each center is a tuple of 
             a random vector of size (the centers) `dim` and an integer k (0 or 1).
    """
    dtype = [('temp', float, dim), ('k', int)]
    centers = np.zeros(len(truth), dtype=dtype)
    for i in range(len(truth)):
        temp = np.random.uniform(0, 1, dim)
        k = 1 if np.random.rand() >= 0.5 else 0
        centers[i] = (temp, k)

    return centers

def get_radius(truth, centers):
    """
    Calculate the radius of each hyper-sphere based on the distance between the center and truth value, 
    adjusted by a small epsilon, based on k. This epislon enforces if the ground truth is within the hyper-sphere (k = 1) or not (k = 0).

    Parameters:
    truth (ndarray): Ground truth values, a 2D array with shape (num_ground_truth, dim).
    centers (ndarray): Centers of the clusters, an array of tuples with the center and k value.

    Returns:
    list: Radii of the clusters, a 1D list of length `num_ground_truth`.
    """
    epsilon = 1e-4
    radius = []
    for i in range(len(truth)):
        curr_center = centers[i][0]
        curr_k = centers[i][1]

        if curr_k == 1:
            radius.append(np.linalg.norm(curr_center - truth[i]) + epsilon)
        else:
            radius.append(np.linalg.norm(curr_center - truth[i]) - epsilon)

    return radius

def get_probability(samples, centers, radius, num_truth):
    """
    This is the Tarp++ algorithm

    Calculates the probability based on the following formula:
    - If k = 1, P = (n + 1) / ((N + 2) * (N + 1))
    - If k = 0, P = (N - n + 1) / ((N + 2) * (N + 1))
    where n is the number of points within the radius, and N is the total number of points.

    Parameters:
    samples (ndarray): Posterior samples from the model, a 4D array of shape (num_model, num_ground_truth, num_samples, dim).
    centers (ndarray): Centers of the clusters, an array of tuples with the center and k value.
    radius (list): Radii of the clusters, a 1D list of length `num_ground_truth`.
    num_truth (int): Number of ground truth samples (num_ground_truth).

    Returns:
    ndarray: Probability of each sample being within its cluster, a 2D array with shape (num_model, num_ground_truth).
    """
    probability = np.zeros((len(samples), num_truth))

    for i in range(num_truth):
        for j in range(len(samples)):
            curr_sample = samples[j, i]
            curr_center = centers[i][0]
            curr_k = centers[i][1]
            curr_radius = radius[i]

            # Distance from every point in curr_sample to curr_center
            distance = np.linalg.norm(curr_sample - curr_center, axis=1)

            # Count how many points are within the radius
            within_circle = distance <= curr_radius
            n = np.sum(within_circle)
            N = len(curr_sample)

            # Calculate probability based on whether k is 0 or 1
            if curr_k == 1:
                curr_prob = (n + 1) / ((N + 2) * (N + 1))
            else:
                curr_prob = (N - n + 1) / ((N + 2) * (N + 1))

            probability[j, i] = curr_prob

    return probability

def tarp_plus_plus(samples, truth, dim):
    """
    Main function to compute the probability of posterior samples belonging to ground truth values, 
    by defining centers and radii for each truth, then calculating probabilities based on distances.

    Parameters:
    samples (ndarray): Posterior samples from the model, a 4D array with shape (num_model, num_ground_truth, num_samples, dim).
    truth (ndarray): Ground truth values, a 2D array with shape (num_ground_truth, dim).
    dim (int): Dimension of the ground truth values.

    Returns:
    ndarray: Probability of each posterior sample being generated from the ground truth values, a 2D array with shape (num_model, num_ground_truth).
    
    Raises:
    ValueError: If `truth` is not a 2D array with the correct shape.
    ValueError: If `samples` does not have the expected 4D shape.
    """
    samples = np.array(samples)
    truth = np.array(truth)

    # Truth must be (n x dim) matrix
    if truth.ndim != 2 or truth.shape[1] != dim:
        raise ValueError(f"truth must be a 2D array with shape (n, {dim})")

    # Samples must be (num_model, num_ground_truth, num_samples, dim) matrix
    if samples.shape[1] != len(truth) or samples.shape[3] != dim:
        raise ValueError(f"samples must be a 4D array with shape (num_model, {len(truth)}, num_samples, {dim})")

    # Define centers and k 
    centers = get_center(truth, dim)

    # Define the radius based on the k value
    radius = get_radius(truth, centers)
        
    # Get the probability of each sample being generated from the ground truth values
    probability = get_probability(samples, centers, radius, len(truth))

    return probability
import torch
from tqdm import tqdm

def get_device():
    """
    Choose the most capable computation device available: CUDA, MPS (Mac GPU), or CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def pokie(truth: torch.Tensor,
          posterior: torch.Tensor,
          num_runs: int = 100,
          device: torch.device = None
          ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Monte Carlo estimation of predictive probabilities, calibration, and per-model normalized counts.

    Returns per-run, per-model n/N values instead of a flattened array.

    Parameters
    ----------
    truth : Tensor of shape (T, q)
        Ground-truth parameter vectors (T samples in q dims).
    posterior : Tensor of shape (M, T, S, q)
        Posterior draws from M models, T truths, S samples each in q dims.
    num_runs : int
        Number of Monte Carlo replications.
    device : torch.device, optional
        Computation device; auto-detected if None.

    Returns
    -------
    avg_prob : Tensor of shape (M,)
        Mean predictive probability per model across runs.
    avg_calibration : Tensor of shape (M,)
        Mean calibration per model across runs.
    n_over_N_vals : Tensor of shape (num_runs, M, T)
        Normalized counts n/N for each run, model, and truth.
    """
    # Device setup
    device = device or get_device()
    truth = truth.to(device)
    posterior = posterior.to(device)

    # Shapes
    M, T, S, q = posterior.shape
    if truth.shape != (T, q):
        raise ValueError(f"Expected truth shape {(T, q)}, got {tuple(truth.shape)}")

    # Constants
    N = S - 1
    max_val = (N + 1) / (N + 2)

    # Pre-allocate
    total_prob = torch.zeros((num_runs, M), device=device)
    total_calib = torch.zeros((num_runs, M), device=device)
    n_over_N_vals = torch.zeros((num_runs, M, T), device=device)

    # Monte Carlo runs
    for run in tqdm(range(num_runs), desc="Pokie MC runs"):
        # 1. Random centers (T, q)
        centers = torch.rand((T, q), device=device)

        # 2. Distances (M, T, S)
        dists = torch.norm(centers[None, :, None, :] - posterior, dim=3)

        # 3. Random radius per (model, truth)
        rand_idx = torch.randint(0, S, (M, T), device=device)
        m_idx = torch.arange(M, device=device)[:, None]
        t_idx = torch.arange(T, device=device)[None, :]
        radii = dists[m_idx, t_idx, rand_idx] + 1e-12

        # 4. Truth distances broadcast (M, T)
        true_dists = torch.norm(centers - truth, dim=1)       # (T,)
        k = (true_dists[None, :] <= radii).float()            # (M, T)

        # 5. Counts per radius (M, T)
        counts = (dists < radii.unsqueeze(2)).sum(dim=2)

        # 6. Predictive probability (M, T)
        prob_in = (counts + 1) / (N + 2)
        prob_out = (N - counts + 1) / (N + 2)
        prob = prob_in * k + prob_out * (1 - k)

        # 7. Calibration (M, T)
        calib = prob / max_val

        # 8. Aggregate
        total_prob[run] = prob.mean(dim=1)
        total_calib[run] = calib.mean(dim=1)

        # 9. Store n/N per model & truth
        n_over_N_vals[run] = counts.float() / N

    # Final averages
    avg_prob = total_prob.mean(dim=0)
    avg_calibration = total_calib.mean(dim=0)

    # Normalize if >1 model
    if M > 1 and avg_prob.sum() > 0:
        avg_prob = avg_prob / avg_prob.sum()

    return avg_prob, avg_calibration, n_over_N_vals


def pokie_with_convergence_estimate(truth: torch.Tensor,
                                    posterior: torch.Tensor,
                                    num_runs: int = 100,
                                    device: torch.device = None
                                    ) -> dict:
    """
    Computes both the empirical Pokie value and the expected theoretical Pokie convergence value
    based on E[lambda] and E[lambda^2] estimated from Monte Carlo.

    Parameters
    ----------
    truth : Tensor (T, q)
    posterior : Tensor (M, T, S, q)
    num_runs : int
    device : torch.device

    Returns
    -------
    result : dict with keys:
        - "empirical_pokie" : (M,)      The Monte Carlo estimate
        - "theoretical_pokie" : (M,)    Estimated P_pokie(M) via E[lambda]
        - "convergence_gap" : (M,)      Difference between theory and empirical
        - "E_lambda" : (M,)             Empirical E[lambda] per model
        - "E_lambda_sq" : (M,)          Empirical E[lambda^2] per model
        - "n_over_N_vals" : (num_runs, M, T)
    """
    # Run standard Pokie
    avg_pokie, avg_calibration, n_over_N = pokie(truth, posterior, num_runs=num_runs, device=device)

    # Compute expectations
    # n_over_N: (num_runs, M, T) --> flatten over (num_runs * T) per model
    M = n_over_N.shape[1]
    N = posterior.shape[2] - 1  # number of posterior samples minus 1

    # n_flat = n_over_N.reshape(num_runs * n_over_N.shape[2], M)  # shape: (num_runs*T, M)
    n_flat = n_over_N.permute(1, 0, 2).reshape(M, -1).T         # shape: (num_runs*T, M)
    E_lambda = n_flat.mean(dim=0)                               # shape: (M,)
    E_lambda_sq = (n_flat ** 2).mean(dim=0)                     # shape: (M,)

    # Theoretical P_pokie(M)
    numer = 2 * N * E_lambda_sq - 2 * N * E_lambda + N + 1
    denom = N + 2
    theo_pokie = numer / denom

    # Compare to empirical
    gap = theo_pokie - avg_calibration

    return {
        "empirical_pokie": avg_pokie,
        "theoretical_pokie": theo_pokie,
        "convergence_gap": gap,
        "E_lambda": E_lambda,
        "E_lambda_sq": E_lambda_sq,
        "n_over_N_vals": n_over_N,
    }


def pokie_bootstrap(truth: torch.Tensor,
                    posterior: torch.Tensor,
                    num_bootstrap: int = 100,
                    num_runs: int = 100,
                    device: torch.device = None
                    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Bootstrap wrapper producing per-bootstrap, per-run, per-model n/N arrays.

    Returns
    -------
    boot_probs : Tensor of shape (num_bootstrap, M)
    boot_calibrations : Tensor of shape (num_bootstrap, M)
    boot_n_over_N : Tensor of shape (num_bootstrap, num_runs, M, T)
        n/N values for each bootstrap, run, model, and truth.
    """
    device = device or get_device()
    truth = truth.to(device)
    posterior = posterior.to(device)

    M, T, S, q = posterior.shape
    boot_probs = torch.zeros((num_bootstrap, M), device=device)
    boot_calib = torch.zeros((num_bootstrap, M), device=device)
    n_values = []

    for b in tqdm(range(num_bootstrap), desc="Bootstrapping pokie"):
        # Resample
        idx = torch.randint(0, T, (T,), device=device)
        truth_bs = truth[idx]
        posterior_bs = posterior[:, idx, :, :]

        # Run pokie
        avg_p, avg_c, n_vals = pokie(
            truth_bs,
            posterior_bs,
            num_runs=num_runs,
            device=device
        )
        boot_probs[b] = avg_p
        boot_calib[b] = avg_c
        n_values.append(n_vals)

    # Stack to (num_bootstrap, num_runs, M, T)
    boot_n_over_N = torch.stack(n_values, dim=0)

    return boot_probs, boot_calib, boot_n_over_N
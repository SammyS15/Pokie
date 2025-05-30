import torch
from tqdm import tqdm
import torch.nn.functional as F
import scipy.spatial.distance as ssd
import numpy as np
from scipy.spatial.distance import cdist

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

# This is temporary that is functionally the same as pokie but allows for different distance metrics 
# Will update to allow the user to add their own distance metric in the future
def pokie_test(truth: torch.Tensor,
               posterior: torch.Tensor,
               num_runs: int = 100,
               distance_metric: str = 'euclidean',
               device: torch.device = None
               ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Monte Carlo estimation of predictive probabilities, calibration, and per‑model normalized counts.
    Automatically picks float32 on MPS (Apple), and default_dtype elsewhere (CPU/CUDA).
    """

    # 1) Device + dtype logic
    device = device or get_device()  # your helper to pick mps/cuda/cpu
    # force float32 on MPS (macOS M1/M2), otherwise use whatever torch.get_default_dtype() is (often float32 or float64)
    chosen_dtype = torch.float32 if device.type == 'mps' else torch.get_default_dtype()

    truth = truth.to(device=device, dtype=chosen_dtype)
    posterior = posterior.to(device=device, dtype=chosen_dtype)

    # 2) Shapes & sanity
    M, T, S, q = posterior.shape
    if truth.shape != (T, q):
        raise ValueError(f"Expected truth shape {(T, q)}, got {tuple(truth.shape)}")

    # 3) Constants
    N = S - 1
    max_val = (N + 1) / (N + 2)

    # 4) Precompute variance vector if needed
    V = None
    if distance_metric == 'seuclidean':
        all_points = posterior.cpu().numpy().astype(np.float32).reshape(-1, q)
        V = np.var(all_points, axis=0, ddof=0)

    # 5) Allocate accumulators (respect dtype)
    total_prob     = torch.zeros((num_runs, M), device=device, dtype=chosen_dtype)
    total_calib    = torch.zeros((num_runs, M), device=device, dtype=chosen_dtype)
    n_over_N_vals  = torch.zeros((num_runs, M, T), device=device, dtype=chosen_dtype)

    # 6) Monte Carlo
    for run in tqdm(range(num_runs), desc="Pokie MC runs"):
        # a) random centers in [0,1)
        centers = torch.rand((T, q), device=device, dtype=chosen_dtype)
        centers_np = centers.cpu().numpy().astype(np.float32)

        # b) compute distances
        if distance_metric == 'euclidean':
            # fast torch-based
            dists = torch.norm(centers[None, :, None, :] - posterior, dim=3)
        else:
            # fallback to SciPy for other metrics
            dists = torch.empty((M, T, S), device=device, dtype=chosen_dtype)
            for m in range(M):
                for t in range(T):
                    post_np = posterior[m, t].cpu().numpy().astype(np.float32)
                    if distance_metric == 'seuclidean':
                        dm = cdist(centers_np[t:t+1], post_np, metric='seuclidean', V=V)
                    else:
                        dm = cdist(centers_np[t:t+1], post_np, metric=distance_metric)
                    dists[m, t] = torch.from_numpy(dm[0].astype(np.float32)).to(device)

        # c) pick a random radius per (model, truth)
        rand_idx = torch.randint(0, S, (M, T), device=device)
        m_idx = torch.arange(M, device=device)[:, None]
        t_idx = torch.arange(T, device=device)[None, :]
        radii = dists[m_idx, t_idx, rand_idx] + 1e-12

        # d) truth-distances & inside‐ball check
        true_dists = torch.norm(centers - truth, dim=1)        # (T,)
        inside = (true_dists[None, :] <= radii).float()        # (M, T)

        # e) counts & predictive probabilities
        counts = (dists < radii.unsqueeze(2)).sum(dim=2)
        prob_in  = (counts + 1) / (N + 2)
        prob_out = (N - counts + 1) / (N + 2)
        prob     = prob_in * inside + prob_out * (1 - inside)
        calib    = prob / max_val

        # f) accumulate
        total_prob[run]    = prob.mean(dim=1)
        total_calib[run]   = calib.mean(dim=1)
        n_over_N_vals[run] = counts.to(dtype=chosen_dtype) / N

    # 7) final averages
    avg_prob = total_prob.mean(dim=0)
    avg_calibration = total_calib.mean(dim=0)

    # 8) normalize across models if >1
    if M > 1 and avg_prob.sum() > 0:
        avg_prob = avg_prob / avg_prob.sum()

    return avg_prob, avg_calibration, n_over_N_vals
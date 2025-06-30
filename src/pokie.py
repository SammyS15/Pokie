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
    quality : Tensor of shape (M,)
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
    total_quality = torch.zeros((num_runs, M), device=device)
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
        total_quality[run] = calib.mean(dim=1)

        # 9. Store n/N per model & truth
        n_over_N_vals[run] = counts.float() / N

    # Average results across runs
    quality = total_quality.mean(dim=0)

    # Normalize quality to [0, 1] range
    quality_norm = (quality - 0.5) / (2/3 - 0.5)
    quality_norm = torch.clamp(quality_norm, min=0.0, max=1.0)
    probabilty = quality_norm / quality_norm.sum() # Normalize probabilities to sum to 1

    return probabilty, quality, n_over_N_vals
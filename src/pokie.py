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
    score : Tensor of shape (M,)
        Pokie Score per model across runs.
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
    total_score = torch.zeros((num_runs, M), device=device)

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
        total_score[run] = calib.mean(dim=1)

    # Average results across runs
    score = total_score.mean(dim=0)

    return score


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
    boot_score : Tensor of shape (num_bootstrap, M)
    """
    device = device or get_device()
    truth = truth.to(device)
    posterior = posterior.to(device)

    M, T, S, q = posterior.shape
    boot_score = torch.zeros((num_bootstrap, M), device=device)

    for b in tqdm(range(num_bootstrap), desc="Bootstrapping pokie"):
        # Resample
        idx = torch.randint(0, T, (T,), device=device)
        truth_bs = truth[idx]
        posterior_bs = posterior[:, idx, :, :]

        # Run pokie
        avg_q, = pokie(
            truth_bs,
            posterior_bs,
            num_runs=num_runs,
            device=device
        )
        boot_score[b] = avg_q

    return boot_score
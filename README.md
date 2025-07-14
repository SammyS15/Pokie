# Pokie: Posterior over k Inference Estimation

Pokie is a Python package for evaluating the calibration and accuracy of posterior distributions through a sample-based, likelihood-free method. It enables Bayesian model comparison in simulation-based settings where the true posterior is unknown and standard metrics fail.

## How Pokie Works

1.	Select a Ground Truth
- Pick a truth sample $y^\*_{j} \sim p(y | x^\*, \mathcal{M}^\*)$
2.	Draw Posterior Samples
- Draw N samples $\{ y_{i,j} \}_{i=1}^N \sim p(y | x^*, \mathcal{M})$
3.	Repeat for Lr Random Regions:
- Sample a random center $c_{j,\ell} \in \mathbb{R}^q$
- Select a random posterior sample to define a radius:
$r_{j,\ell} = \| c_{j,\ell} - y_{i,j} \|$
- Define a region $\mathcal{R}{j,\ell}$ as the hypersphere around $c_{j,\ell}$ with radius $r_{j,\ell}$
4.	Count Points
- Count how many posterior samples fall into the region: $n = \sum_i \mathbf{1}[y_{i,j} \in \mathcal{R}_{j,\ell}]$
- Check if the ground-truth sample falls inside: $k = \mathbf{1}[y^*j \in \mathcal{R}_{j,\ell}]$
5.	Update Score
```
if k == 1:
    score += (n + 1) / (N + 2)
else:
    score += (N - n + 1) / (N + 2)
```
6.	Final Pokie Score:
$P_{\text{Pokie}}(\mathcal{M}) = \frac{\texttt{score}}{L \cdot L_r}$


## Interpretation
- Well-calibrated posterior → Pokie score → ≈ 2/3
- Misaligned posterior → Pokie score → ≈ 1/2

## Example Usage
```
import torch
from pokie import pokie

# Ground truth parameters (T samples in q-dim)
truth = torch.randn(T, q)

# Posterior samples from M models (M, T, S, q)
posterior = torch.randn(M, T, S, q)

# Run Pokie
score = pokie(truth, posterior, num_runs=100)
print(score)  # Pokie scores for each model
```

## Developing

If you're a developer then:

```python
git clone git@github.com:SammyS15/Pokie.git
cd Pokie
git checkout -b my-new-branch
pip install -e .
```

But make an issue first so we can discuss implementation ideas.

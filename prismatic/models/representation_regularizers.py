"""Losses that keep a trainable visual encoder from quietly narrowing.

Once the encoder is unfrozen it produces both the world model's input and, through
the EMA target, its own regression objective. EMA prevents the pair from jumping to
a constant, but not from drifting together into an ever smaller subspace over tens
of thousands of steps -- which shows up as falling per-dimension variance and rising
similarity between patches that ought to differ, while every loss keeps improving.

These two terms act directly on those quantities, in the spirit of VICReg:

* **variance**: a hinge that pushes each latent channel's standard deviation up to a
  floor. It only acts on channels that have gone quiet, so a healthy encoder pays
  nothing.
* **covariance**: penalises off-diagonal correlation, pushing channels to carry
  different information rather than redundant copies of the same few directions.

Both are computed in fp32 regardless of the compute dtype: they are statistics of
small magnitude, and bf16 rounding would swamp the signal near the hinge.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def variance_covariance_regularization(
    features: torch.Tensor,
    *,
    variance_floor: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (variance hinge, covariance penalty) for `features` [..., D].

    Every leading dimension is flattened into the sample axis, so patches and batch
    entries count alike -- the diagnostic that motivated this measures spread across
    patches, and collapse shows up there first.
    """
    x = features.float().reshape(-1, features.shape[-1])
    if x.shape[0] < 2:
        zero = x.sum() * 0.0
        return zero, zero

    centered = x - x.mean(dim=0, keepdim=True)

    # Hinge, not a plain penalty: pushing variance up without bound would just
    # inflate the scale of the representation rather than restore its diversity.
    std = torch.sqrt(centered.var(dim=0) + eps)
    variance = torch.relu(variance_floor - std).mean()

    covariance_matrix = (centered.T @ centered) / (x.shape[0] - 1)
    off_diagonal = covariance_matrix - torch.diag_embed(torch.diagonal(covariance_matrix))
    covariance = off_diagonal.pow(2).sum() / features.shape[-1]

    return variance, covariance


def sketched_isotropic_gaussian_regularization(
    features: torch.Tensor,
    *,
    num_directions: int = 64,
    num_frequencies: int = 16,
    max_frequency: float = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SIGReg: push the embedding distribution toward an isotropic Gaussian.

    LeJEPA's result is that the embedding distribution minimising downstream
    prediction risk is isotropic Gaussian, which makes the target distribution
    unique rather than a hand-tuned set of statistics. Matching it in D dimensions
    is intractable, so the distribution is *sketched*: project onto random unit
    directions and run a one-dimensional goodness-of-fit test against N(0,1) on
    each slice. A distribution is isotropic Gaussian exactly when every such
    projection is N(0,1), so the sketch loses nothing in the limit.

    The per-slice distance is the Epps-Pulley statistic -- the weighted squared
    distance between the empirical characteristic function and the normal's,
    ``exp(-t^2/2)`` -- evaluated on a fixed frequency grid. It is built from
    sines and cosines, so its gradients are bounded no matter how far the current
    distribution is from the target, and it costs O(N * D * R) with no covariance
    matrix to form or invert.

    Unlike a variance/covariance pair, this constrains the *shape* of each slice,
    so cluster collapse -- several tight modes whose per-channel variance and
    decorrelation both look healthy -- is penalised too.
    """
    x = features.float().reshape(-1, features.shape[-1])
    if x.shape[0] < 2:
        return x.sum() * 0.0

    # Centre only. Rescaling to unit norm -- globally or per-direction -- would
    # make the test blind to exactly the failures it exists to catch: a constant
    # embedding rescales into indistinguishable noise, and per-direction scaling
    # hides anisotropy. N(0, I) is the target, so the scale is part of it.
    centered = x - x.mean(dim=0, keepdim=True)

    directions = torch.randn(
        x.shape[-1], num_directions, device=x.device, dtype=x.dtype, generator=generator
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-6)
    projections = centered @ directions                       # [N, R]

    frequencies = torch.linspace(
        max_frequency / num_frequencies, max_frequency, num_frequencies,
        device=x.device, dtype=x.dtype,
    )
    angles = projections[..., None] * frequencies             # [N, R, F]
    real = angles.cos().mean(dim=0) - (-0.5 * frequencies.square()).exp()
    imaginary = angles.sin().mean(dim=0)
    # Gaussian weighting keeps the high frequencies, which the empirical
    # characteristic function estimates worst, from dominating the statistic.
    weights = (-0.5 * frequencies.square()).exp()
    return ((real.square() + imaginary.square()) * weights).sum(dim=-1).mean()


def gather_across_ranks(features: torch.Tensor) -> torch.Tensor:
    """Concatenate every rank's rows, keeping this rank's gradient.

    A distributional constraint needs samples. Per device there are eight, and a
    pooled latent has 256 dimensions, so a per-rank estimate is worthless -- the
    reason the scale had to be pinned by construction rather than by a regularizer
    in the first place. Gathering sixteen ranks turns eight rows into hundreds.
    Other ranks arrive detached: their gradients flow on their own devices, and
    double-counting them here would scale the update by the world size.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return features
    world_size = dist.get_world_size()
    if world_size <= 1:
        return features
    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.detach().contiguous())
    gathered[dist.get_rank()] = features
    return torch.cat(gathered, dim=0)


def temporal_separation_hinge(
    current: torch.Tensor,
    future: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    """Penalise the latent for forgetting that time passed.

    Measured directly, because measuring it indirectly does not work. Pinning the
    latent's norm stops it shrinking, and a distributional test on current and
    future latents stacked together is nearly blind to this: collapsing time only
    reduces the number of *distinct* rows, while the marginal stays healthy on the
    strength of the differences between samples -- on synthetic data a fully
    time-collapsed stack scored just twice a healthy one, the same weak signal
    that let a variance-conserving dimensional collapse through earlier.

    The quantity that actually failed is the size of ``z_{t+k} - z_t``, which fell
    0.105 -> 0.015 over 750 steps. So that is what is bounded, with a hinge: a
    latent that keeps its horizons apart pays nothing.
    """
    # One pooled token per timestep gives [B,D] and [B,K,D]; several give
    # [B,Q,D] and [B,K,Q,D]. The horizon axis is always the one inserted.
    if current.ndim not in (2, 3) or future.ndim != current.ndim + 1:
        raise ValueError(
            "expected [B,D] / [B,K,D] or [B,Q,D] / [B,K,Q,D] current and future latents"
        )
    separation = future.float() - current.float()[:, None]
    rms = separation.pow(2).mean().sqrt()
    return torch.relu(torch.as_tensor(float(floor), device=rms.device, dtype=rms.dtype) - rms)


def query_diversity_penalty(latents: torch.Tensor) -> torch.Tensor:
    """Penalise the pooling queries for all summarising the same thing.

    The distributional terms above flatten ``[B,T,Q,D]`` into ``D``-wide rows and
    score them as independent samples, so ``Q`` queries emitting one identical
    vector reads to them as ``Q`` duplicate rows -- which changes neither the
    per-dimension variance nor the between-dimension covariance. They are blind to
    this failure by construction, and it is the one that happened: at step 7000 the
    sixteen queries' mean off-diagonal cosine was 0.99993 and the pooled latent's
    effective rank was 1.05 of a possible 16, while the encoder feeding it held
    733 and the raw patch motion 14.99 of a possible 16. The pooler, not the data
    and not the world model, was where the signal went.

    So this measures the axis they cannot see: the mean cosine between distinct
    queries of the same sample, centred per query across the batch first. Without
    centring, the direction every clip shares would count as agreement and a
    healthy pooler could never score near zero.

    Returns a value in ``[0, 1]``-ish: 1.0 when every query is identical, ~0 when
    they are mutually orthogonal. Queries are not pushed to be anti-correlated --
    the penalty is on the mean cosine, whose floor for `Q` centred unit vectors is
    ``-1/(Q-1)``, not on its magnitude.
    """
    if latents.ndim < 3:
        raise ValueError("expected pooled latents shaped [B,Q,D] or [B,T,Q,D]")
    flat = latents.float().reshape(-1, latents.shape[-2], latents.shape[-1])
    queries = flat.shape[1]
    if queries < 2:
        return flat.new_zeros(())
    # Centre each query across the batch: the component every sample shares is the
    # task's common direction, not evidence that two queries do the same job.
    centred = flat - flat.mean(dim=0, keepdim=True)
    normed = centred / centred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gram = torch.einsum("bqd,bpd->bqp", normed, normed)
    off_diagonal = ~torch.eye(queries, dtype=torch.bool, device=gram.device)
    return gram[:, off_diagonal].mean()


@torch.no_grad()
def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """How many directions the features actually occupy, as a diagnostic only.

    ``exp`` of the entropy of the covariance's normalised eigenvalue spectrum: `D`
    when the features fill their space, 1.0 when they lie on a single line. This is
    the number that made the pooled collapse legible -- 1.05 of a possible 16 while
    every loss term in the log read healthy -- and it was only ever computed
    offline. Logged every step so the next such collapse is visible while it is
    happening rather than a thousand steps later.

    Eigenvalues of the `D x D` covariance rather than an SVD of the `N x D` batch:
    same spectrum, and `D` here is 256 while `N` is several thousand.
    """
    flat = features.float().reshape(-1, features.shape[-1])
    if flat.shape[0] < 2:
        return flat.new_zeros(())
    centred = flat - flat.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / (centred.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
    spectrum = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    spectrum = spectrum[spectrum > 0]
    return torch.exp(-(spectrum * spectrum.log()).sum()).to(features.dtype)

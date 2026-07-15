#!/usr/bin/env python
"""Generic post-hoc component pruning for a bundled VSA memory -- numpy/JAX port
of examples/learnable_points/vsa_pruning.py (torch) for callers that already
work in JAX, e.g. scripts/room_id_scale_modulation.py. See vsa_pruning.md for
the full mathematical definition; short version: given a memory M in C^D (or
R^D) whose retrieval score is a linear functional of M, pruning to a D'-subset
via mu in {0,1}^D by zeroing (tilde_M = mu * M) is exactly equivalent to
slicing the model to D' dimensions, up to a global constant absorbed by
whatever calibration sits downstream.

Three selection strategies (random is the null-hypothesis baseline, included
for completeness even though callers may only want magnitude/priority/gate):
  random     -- uniform random D'-subset
  magnitude  -- keep the D' components with the largest caller-supplied score
  priority   -- keep the D' smallest/largest of a caller-supplied fixed score
  gate       -- per-component sigmoid gate fit against a caller-supplied
                differentiable loss, progressive across shrinking budgets so
                masks nest by construction

This module owns only mask arithmetic; the caller supplies every score/loss.
Unlike the torch original, one component index (`dc_index`, typically 0 for a
half-spectrum's DC bin) can be pinned always-on -- callers whose downstream
code has no path for zeroing that component (e.g. every gain vector in
room_id_scale_modulation.py hardcodes weight 1 there) should always pass it.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------

def apply_mask(memory: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """tilde_M = mu (*) M -- zero every component outside the mask."""
    return memory * mask


def retained_energy(component: np.ndarray, mask: np.ndarray,
                     bin_weight: Optional[np.ndarray] = None) -> float:
    """sum_{b in supp(mu)} bin_weight[b] |component_b|^2 -- the fraction of a
    half-spectrum's squared real-domain norm surviving the mask (1.0 if
    component is unit-normalized in that sense and mask keeps everything).
    `bin_weight` defaults to the conjugate-pair convention ([1, 2, 2, ...])
    used throughout room_id_scale_modulation.py's half-spectrum machinery."""
    n = component.shape[-1]
    if bin_weight is None:
        bin_weight = np.where(np.arange(n) == 0, 1.0, 2.0)
    return float(np.sum(bin_weight * np.abs(component) ** 2 * mask))


def _rank_order(score: np.ndarray, descending: bool, dc_index: Optional[int]) -> np.ndarray:
    """Indices other than `dc_index`, ranked by `score` (descending or
    ascending), best first."""
    n = score.shape[0]
    pool = np.delete(np.arange(n), dc_index) if dc_index is not None else np.arange(n)
    order = np.argsort(score[pool])
    if descending:
        order = order[::-1]
    return pool[order]


def _mask_from_order(order: np.ndarray, D_prime: int, n: int, dc_index: Optional[int]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    k = D_prime
    if dc_index is not None:
        mask[dc_index] = True
        k = D_prime - 1
    mask[order[:max(k, 0)]] = True
    return mask


# ----------------------------------------------------------------------
# Strategies: random / magnitude / priority
# ----------------------------------------------------------------------

def random_mask(n_bins: int, D_prime: int, rng: Optional[np.random.Generator] = None,
                 dc_index: Optional[int] = 0) -> np.ndarray:
    """Uniform random D'-subset of {0, ..., n_bins-1} (dc_index forced in if
    given). Each call draws an independent subset -- unlike magnitude/priority/
    gate, this does not nest across successive D'."""
    rng = rng or np.random.default_rng()
    pool = np.delete(np.arange(n_bins), dc_index) if dc_index is not None else np.arange(n_bins)
    k = D_prime - 1 if dc_index is not None else D_prime
    idx = rng.permutation(pool)[:max(k, 0)]
    mask = np.zeros(n_bins, dtype=bool)
    if dc_index is not None:
        mask[dc_index] = True
    mask[idx] = True
    return mask


def magnitude_mask(score: np.ndarray, D_prime: int, dc_index: Optional[int] = 0) -> np.ndarray:
    """Keep the D' components with the largest caller-supplied score (e.g. a
    bin's |M_hat| or a per-object energy aggregate). Data-dependent: a pure
    function of whatever score the caller computed from the trained memory."""
    order = _rank_order(score, descending=True, dc_index=dc_index)
    return _mask_from_order(order, D_prime, score.shape[0], dc_index)


def priority_mask(priority: np.ndarray, D_prime: int, ascending: bool = True,
                   dc_index: Optional[int] = 0) -> np.ndarray:
    """Keep the D' components with the smallest (ascending=True) or largest
    (ascending=False) value of a caller-supplied, memory-blind priority score
    (e.g. a bin's fixed scale/frequency)."""
    order = _rank_order(priority, descending=not ascending, dc_index=dc_index)
    return _mask_from_order(order, D_prime, priority.shape[0], dc_index)


def ranking_curve(score: np.ndarray, dprimes: List[int], descending: bool = True,
                   dc_index: Optional[int] = 0) -> Dict[int, np.ndarray]:
    """One argsort, sliced at every D' in `dprimes` -- avoids repeating the
    O(n log n) sort magnitude_mask/priority_mask would each do per D'."""
    order = _rank_order(score, descending=descending, dc_index=dc_index)
    n = score.shape[0]
    return {dp: _mask_from_order(order, dp, n, dc_index) for dp in dprimes}


# ----------------------------------------------------------------------
# Strategy: learned gate
# ----------------------------------------------------------------------

@dataclass
class GateFitResult:
    """One gate-fitting stage's outcome."""
    logits: np.ndarray          # [n_bins], -inf outside support, +inf at dc_index
    loss_history: np.ndarray


def fit_gate(n_bins: int, support: np.ndarray, loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
             steps: int = 150, lr: float = 0.05, lam: float = 1e-3,
             dc_index: Optional[int] = 0) -> GateFitResult:
    r"""Fine-tune a per-component sigmoid gate sigma(phi) in [0,1]^n_bins,
    restricted to `support` (bool [n_bins]):

        g   = mu_support (*) sigma(phi), with g[dc_index] pinned to 1
        loss = loss_fn(g) + lam * mean_{b in support} sigma(phi_b)

    `loss_fn` is entirely caller-supplied and differentiable w.r.t. its
    (n_bins,) input -- this is how all domain-specific behavior (which D
    tensor, which targets, how to batch/resample) enters; this function only
    knows how to gate and regularize toward closed. Components outside
    `support` get exactly zero gradient (g is 0 there regardless of phi_b),
    so only phi_b for b in support ever moves. Returned logits are -inf
    outside support (so a later topk can never reselect a dropped bin) and
    +inf at dc_index (so it is always reselected).
    """
    support_j = jnp.asarray(support)
    support_f = support_j.astype(jnp.float32)
    logit0 = jnp.zeros(n_bins)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(logit0)

    def gate_from_logit(logit):
        g = support_f * jax.nn.sigmoid(logit)
        if dc_index is not None:
            g = g.at[dc_index].set(1.0)
        return g

    def loss_full(logit):
        g = gate_from_logit(logit)
        denom = jnp.maximum(support_f.sum(), 1.0)
        reg = jnp.sum(jax.nn.sigmoid(logit) * support_f) / denom
        return loss_fn(g) + lam * reg

    @jax.jit
    def step(logit, opt_state):
        loss, grads = jax.value_and_grad(loss_full)(logit)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(logit, updates), opt_state, loss

    logit = logit0
    history = np.empty(steps)
    for i in range(steps):
        logit, opt_state, loss = step(logit, opt_state)
        history[i] = float(loss)

    out = np.array(logit)
    out[~np.asarray(support)] = -np.inf
    if dc_index is not None:
        out[dc_index] = np.inf
    return GateFitResult(logits=out, loss_history=history)


def progressive_gate_masks(n_bins: int, dprimes: List[int],
                           loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
                           steps: int = 150, lr: float = 0.05, lam: float = 1e-3,
                           dc_index: Optional[int] = 0,
                           ) -> Tuple[Dict[int, np.ndarray], Dict[int, GateFitResult]]:
    r"""Iterative pruning: for each successively smaller D' in `dprimes`
    (should be strictly decreasing), fit_gate a fresh gate restricted to the
    *previous* stage's survivors, then keep its top-D' logits as the next
    support. Masks nest by construction because each stage can only narrow
    `support`, never reintroduce a dropped bin (dead logits pinned to -inf;
    dc_index pinned to +inf so it always survives)."""
    support = np.ones(n_bins, dtype=bool)
    masks, results = {}, {}
    for dp in dprimes:
        result = fit_gate(n_bins, support, loss_fn, steps=steps, lr=lr, lam=lam, dc_index=dc_index)
        keep = np.argsort(result.logits)[::-1][:dp]
        support = np.zeros(n_bins, dtype=bool)
        support[keep] = True
        masks[dp] = support.copy()
        results[dp] = result
    return masks, results

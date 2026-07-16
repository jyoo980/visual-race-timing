"""Timing prior for crossing identification.

At the instant a runner crosses,
their *lap history* says a lot about whether it is plausibly them right now. A
runner who crossed 10 s ago is almost certainly not the one crossing now; a
runner whose typical lap is 90 s and who last crossed ~90 s ago is very likely.

This module builds, per runner, a causal predictive model of "am I due to
cross at time t?" and returns a log-likelihood we can fuse with the appearance
score. "Causal" = at query time ``t`` we only ever take into account that runner's
crossings strictly before ``t``.

Model
-----
For runner ``r`` at realtime ``t`` (seconds):

* ``last`` = most recent crossing < t (or the runner's wave start if none).
* ``elapsed = t - last``.
* Lap-duration estimate ``mu_r`` from consecutive crossing gaps before ``t``
  (EWMA, so recent laps dominate), shrunk toward a
  pooled prior ``mu0`` (typical lap time for this event's field) when the runner has few laps. ``sigma_r`` likewise.
* Because crossings can be missed (unlabeled), model the elapsed gap as a sum
  of ``k`` laps and marginalize with a geometric miss-penalty ``rho``:

      P_time(t|r) = sum_{k=1..K} rho^{k-1} * N(elapsed; k*mu_r, k*sigma_r^2)

  k=1 is the normal "due for next lap" case; k>=2 absorbs missed crossings.
  When ``elapsed << mu_r`` (just crossed) every term is ~0 -> r is suppressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Tuned defaults. w_t is the weight on the (scale-normalized) timing penalty;
DEFAULT_TAU_A = 0.1
DEFAULT_W_T = 1.0


NORM_WINSOR = 15.0    # nats: candidates worse than (best − this) are all "ruled out"
NORM_MIN_STD = 0.5    # nats: below this the set is uninformative → pure appearance
NORM_CLIP = 2.0       # spreads: deepest timing penalty any one candidate may take


def _logsumexp(values: Sequence[float]) -> float:
    vals = [v for v in values if v != -math.inf]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _timing_penalty(log_time: Sequence[float]) -> List[float]:
    """How much each candidate's timing evidence hurts it relative to the
    best-timed candidate in this set, in standard deviations.

    Converts raw timing log-likelihoods into a relative score: the best-timed
    candidate gets ``0`` (no penalty), everyone else gets a negative number.

    * Winsorize each ``lp_i`` at ``max − NORM_WINSOR`` so a "just crossed" outlier
      (lp ≈ −20) can't inflate the spread and crush the distinctions among the
      genuinely plausible candidates.
    * Standardize by the post-winsorize std, so the penalty is measured in
      spreads rather than raw nats.
    * Clip at ``−NORM_CLIP`` so no candidate loses more than a bounded amount of
      timing evidence (raw log-density tails are the least-calibrated part of the
      lap model; rank + relative spread are the trustworthy content).

    Returns all-zeros (fusion reduces to pure appearance) when timing is
    uninformative: fewer than two *finite* log-likelihoods, or a winsorized
    spread below ``NORM_MIN_STD`` (e.g. everyone on schedule). ``-inf`` entries
    (ruled-out candidates) are winsorized up to ``max − NORM_WINSOR`` and so
    land in the worst penalty bucket.
    """
    lp = np.asarray(log_time, dtype=np.float64)
    finite = lp[np.isfinite(lp)]
    if finite.size < 2:
        return [0.0] * lp.size
    m = float(finite.max())
    lpw = np.maximum(lp, m - NORM_WINSOR)   # -inf clamps to m − NORM_WINSOR
    s = float(lpw.std())
    if s < NORM_MIN_STD:
        return [0.0] * lp.size
    z = np.maximum((lpw - m) / s, -NORM_CLIP)
    return z.tolist()


def _log_normal_pdf(x: float, mean: float, var: float) -> float:
    if var <= 0:
        return -math.inf
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mean) ** 2 / var)


@dataclass
class TimingParams:
    """Hyperparameters for the timing prior.

    Defaults are the production configuration (``annotate.py`` builds a
    ``TimingPrior`` with ``TimingParams()``).
    """
    ewma_alpha: float = 0.6          # weight on the *running* estimate vs newest lap
    min_laps_for_own: int = 2        # need this many laps before trusting a runner's own mu
    shrink_k: float = 2.0            # pseudo-count pulling mu_r toward mu0 (Bayesian shrinkage)
    sigma_floor_frac: float = 0.08   # sigma >= this fraction of mu (never absurdly confident)
    cold_sigma_inflate: float = 1.5  # widen sigma when falling back to the pooled prior
    k_max: int = 3                   # max laps to marginalize over (missed crossings)
    rho: float = 0.2                 # geometric penalty per missed crossing
    trim_lo_pct: float = 5.0         # robust pooled-stats trimming (drop stops/sprints)
    trim_hi_pct: float = 90.0


@dataclass
class RunnerHistory:
    runner_id: int
    crossings: np.ndarray            # realtime seconds, sorted ascending (ALL crossings)
    start_time: Optional[float]      # wave start, realtime seconds (or None)


@dataclass
class TimingPrior:
    """Per-runner causal timing likelihood over a fixed candidate universe."""

    histories: Dict[int, RunnerHistory]
    params: TimingParams = field(default_factory=TimingParams)
    mu0: float = 0.0                 # pooled lap mean (robust)
    sigma0: float = 0.0              # pooled lap std (robust)

    @classmethod
    def build(cls, crossings_by_runner: Dict[int, Sequence[float]],
              start_by_runner: Dict[int, float],
              params: Optional[TimingParams] = None) -> "TimingPrior":
        params = params or TimingParams()
        histories: Dict[int, RunnerHistory] = {}
        all_laps: List[float] = []
        for rid, times in crossings_by_runner.items():
            arr = np.sort(np.asarray(times, dtype=np.float64))
            histories[rid] = RunnerHistory(rid, arr, start_by_runner.get(rid))
            if arr.size >= 2:
                all_laps.extend(np.diff(arr).tolist())
        # Seed runners who have a wave start but no crossings yet, so their first
        # (entrance) lap is still scored against elapsed-since-start.
        for rid, start in start_by_runner.items():
            if rid not in histories:
                histories[rid] = RunnerHistory(rid, np.empty((0,), dtype=np.float64), start)
        mu0, sigma0 = cls._robust_pooled(all_laps, params)
        return cls(histories=histories, params=params, mu0=mu0, sigma0=sigma0)

    @staticmethod
    def _robust_pooled(laps: List[float], params: TimingParams):
        if not laps:
            # No data at all: neutral fallback (won't matter, prior is uniform then).
            return 90.0, 30.0
        a = np.asarray(laps, dtype=np.float64)
        # Asymmetric percentile trim (drops stops/sprints + some missed-crossing doubles).
        lo = np.percentile(a, params.trim_lo_pct)
        hi = np.percentile(a, params.trim_hi_pct)
        kept = a[(a >= lo) & (a <= hi)]
        if kept.size == 0:
            kept = a
        mu0 = float(np.mean(kept))
        sigma0 = float(np.std(kept))
        if sigma0 <= 0:
            sigma0 = max(1.0, params.sigma_floor_frac * mu0)
        return mu0, sigma0

    @staticmethod
    def _fold_missed_crossings(laps: np.ndarray) -> np.ndarray:
        """Fold missed-crossing "doubles" back to single-lap estimates.

        The likelihood asserts a miss rate ``rho`` > 0, so ~``rho`` of a runner's
        *own* observed gaps span k>1 real laps (a gap of ~k·median). Feeding those
        raw into the EWMA/std would inflate both mu and sigma, contradicting the
        thinning model (the pooled estimator already trims them; the per-runner
        path must too, or the runners with the most history -- where the per-runner
        estimate is supposed to win -- get the worst-biased mu/sigma). Using the
        runner's own median gap as the single-lap scale, read each gap as
        ``k = round(gap/median)`` laps and divide it back to a per-lap estimate.
        """
        if laps.size == 0:
            return laps
        med = float(np.median(laps))
        if med <= 0:
            return laps
        k = np.maximum(1.0, np.round(laps / med))
        return laps / k

    def _runner_mu_sigma(self, laps: np.ndarray):
        """Predictive lap mean/std from a runner's own laps (already causal)."""
        p = self.params
        if laps.size >= p.min_laps_for_own:
            laps = self._fold_missed_crossings(laps)
            # EWMA in chronological order: newest laps weighted most (fatigue drift).
            mu = laps[0]
            for d in laps[1:]:
                mu = p.ewma_alpha * mu + (1.0 - p.ewma_alpha) * d
            # Shrink toward the pooled mean by pseudo-count.
            n = laps.size
            mu = (n * mu + p.shrink_k * self.mu0) / (n + p.shrink_k)
            sigma = float(np.std(laps))
            sigma = max(sigma, p.sigma_floor_frac * mu)
            # Blend dispersion toward pooled too when few laps.
            sigma = (n * sigma + p.shrink_k * self.sigma0) / (n + p.shrink_k)
            return mu, sigma
        # Cold start: pooled prior with inflated variance.
        return self.mu0, self.sigma0 * p.cold_sigma_inflate

    def _uninformative_loglik(self) -> float:
        """Log-density for a runner we can't place in time (unknown to the model,
        or has no crossings and no wave-start anchor -> no ``elapsed`` to score).

        Return the *peak* density of the coldest lap model we ever use (pooled
        sigma inflated by ``cold_sigma_inflate``): such a runner is then as
        plausible as an on-schedule cold-start runner -- it ties with, never
        dominates, real candidates. A flat ``0.0`` would exceed any achievable
        term and, since ``_timing_penalty`` standardizes against the max, would
        silently make the unknown runner the timing favorite.
        """
        sigma = max(self.sigma0, 1.0) * self.params.cold_sigma_inflate
        return -0.5 * math.log(2.0 * math.pi * sigma * sigma)

    def log_likelihood(self, runner_id: int, t: float) -> float:
        """log P_time(t | runner crosses now), using only crossings < t.

        Returns a neutral density (``_uninformative_loglik``) for
        a runner with no history and no known start, and ``-inf`` for the "ruled
        out" degenerate cases (query at/behind the last crossing). Both are
        handled downstream by the candidate-set standardization in
        ``_timing_penalty`` (which winsorizes the tail), so there is no separate
        absolute likelihood floor here.
        """
        hist = self.histories.get(runner_id)
        if hist is None:
            return self._uninformative_loglik()
        p = self.params
        crossings = hist.crossings
        before = crossings[crossings < t]
        if before.size:
            last = float(before[-1])
            laps = np.diff(before)
        elif hist.start_time is not None:
            last = float(hist.start_time)
            laps = np.empty((0,), dtype=np.float64)
        else:
            return self._uninformative_loglik()  # no anchor -> non-informative

        elapsed = t - last
        if elapsed <= 0:
            # Query at/behind last known crossing: implausible to cross again now.
            return -math.inf

        mu, sigma = self._runner_mu_sigma(laps)
        if mu <= 0:
            return -math.inf
        var1 = sigma * sigma

        terms = []
        for k in range(1, p.k_max + 1):
            if p.rho > 0:
                k_penalty = (k - 1) * math.log(p.rho)
            else:
                k_penalty = 0.0 if k == 1 else -math.inf
            terms.append(k_penalty + _log_normal_pdf(elapsed, k * mu, k * var1))
        return _logsumexp(terms)


def fuse_ranking(prior: TimingPrior, candidate_ids: Sequence[int],
                  appearance_dists: Sequence[float], t: float,
                  tau_a: float = DEFAULT_TAU_A, w_t: float = DEFAULT_W_T):
    """Fuse appearance cosine distances with ``prior``'s timing likelihood at time ``t``.

    ``candidate_ids`` and ``appearance_dists`` are parallel (same order boxmot
    returns). Returns ``(ranked_ids, ranked_app_dists, posterior)`` sorted
    best-first, where ``posterior`` is the softmax weight of each ranked
    candidate under the fused score (useful for an auto-accept gate). The
    returned distances are the *appearance* cosine distances (so the human
    still sees the appearance confidence); only the order reflects timing.

    The timing term is standardized to the spread *within this candidate set*
    (see ``_timing_penalty``).
    """
    ids = [int(i) for i in candidate_ids]
    dists = [float(d) for d in appearance_dists]
    penalty = _timing_penalty([prior.log_likelihood(i, t) for i in ids])
    scores = [(-d / tau_a) + w_t * z
              for d, z in zip(dists, penalty)]
    order = sorted(range(len(ids)), key=lambda j: scores[j], reverse=True)
    ranked_ids = [ids[j] for j in order]
    ranked_dists = [dists[j] for j in order]
    ranked_scores = [scores[j] for j in order]
    # Softmax posterior over the (finite) fused scores.
    finite = [s for s in ranked_scores if s != -math.inf]
    if finite:
        m = max(finite)
        exps = [math.exp(s - m) if s != -math.inf else 0.0 for s in ranked_scores]
        z_sum = sum(exps) or 1.0
        posterior = [e / z_sum for e in exps]
    else:
        posterior = [0.0] * len(ranked_ids)
    return ranked_ids, ranked_dists, posterior
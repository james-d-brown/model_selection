"""
Linear tier: linear-regression residual test (partial canonical correlation).

Z is removed from the candidate and the observations by least squares, and
the residuals' canonical correlations are summarised as one minus Wilks'
lambda (Hotelling, 1936), the squared partial correlation for scalars.

Under H0 every product of a candidate residual and an observation residual
has zero conditional mean, so the null distribution is built by a
multiplier (wild) bootstrap of those products, as in the nonlinear tier: each draw
reweights the per-sample products by multipliers of mean zero and variance
one and recomputes the statistic, without moving either series. Unlike a
permutation of residuals, which assumes they are exchangeable, the
multiplier bootstrap reproduces the variance of the products when the
residuals' spread varies with Z or with each other (Wu, 1986; Mammen, 1993),
which permutation does not (Winkler et al., 2014). Samples closer than the
dependence lag receive correlated multipliers (the dependent wild bootstrap,
Shao, 2010); on evenly spaced samples the bandwidth is the Politis–White
estimate for the products of the candidate and observation residuals. Each
product is divided by 1 - h_t, with h_t the leverage of sample t in the
regression on Z, because least-squares residuals understate the errors'
variance there (the HC2 correction, MacKinnon and White, 1985; Davidson and
Flachaire, 2008). All candidates share each set of multipliers.

See docs/methods.md, Linear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.association.canonical import canonical_correlations
from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights
from model_selection.statistics.regression.linear import ols_fit
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block
from model_selection.utils import as2d


def partial_canonical_correlation(resid_c: np.ndarray, resid_o: np.ndarray) -> float:
    """1 - prod(1 - rho_i^2) over the canonical correlations of two residual
    matrices (already centred)."""
    rhos = canonical_correlations(resid_c, resid_o)
    return float(1.0 - np.prod(1.0 - rhos ** 2))


def _orthonormal_columns(R: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the column space of the centred residual
    matrix R, to numerical rank."""
    U, sv, _ = np.linalg.svd(as2d(R), full_matrices=False)
    if sv.size == 0 or sv[0] <= 0:
        return U[:, :0]
    return U[:, sv > sv[0] * max(R.shape) * np.finfo(float).eps]


def _set_correlation(M: np.ndarray) -> np.ndarray:
    """1 - prod(1 - s_i^2) over the singular values s_i of each matrix in the
    stack M (..., p, q): the set correlation of two orthonormal bases with
    cross-product M. Singular values are capped just below one."""
    s = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0 - 1e-12)
    return 1.0 - np.prod(1.0 - s ** 2, axis=-1)


MULTIPLIER_CHUNK = 64  # draws generated at a time, to bound memory on long records


class LinearResidualTest(ConditionalIndependenceTest):
    """Partial canonical correlation with a multiplier-bootstrap null.

    run_length : consecutive insignificant autocorrelations that end the
        decorrelation-lag search (docs/methods.md, Dependence between samples).
    """
    name = "linear"

    def __init__(self, run_length: int = DEFAULT_RUN_LENGTH):
        self.run_length = run_length

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        names = list(candidates)
        n = as2d(obs).shape[0]
        fitted_o, resid_o = ols_fit(obs, conditioning)
        resid_c = {}
        diagnostics = {}
        tss_o = float(np.sum((as2d(obs) - as2d(obs).mean(0)) ** 2))
        r2_o = 1.0 - float(np.sum(resid_o ** 2)) / tss_o if tss_o > 0 else 0.0
        k = 0 if conditioning is None else as2d(conditioning).shape[1]
        for name in names:
            c = as2d(candidates[name])
            fitted_c, r = ols_fit(c, conditioning)
            resid_c[name] = r
            tss_c = float(np.sum((c - c.mean(0)) ** 2))
            diagnostics[name] = {
                "r2_candidate_on_selected": 1.0 - float(np.sum(r ** 2)) / tss_c if tss_c > 0 else 0.0,
                "r2_obs_on_selected": r2_o,
                "samples_per_conditioning_dimension": n / max(k, 1),
            }
        observed = np.array([partial_canonical_correlation(resid_c[nm], resid_o) for nm in names])
        resolved = resolve_block(time_positions, dependence_lag, [(resid_c[nm], resid_o) for nm in names],
                                 self.run_length, grid_step)
        block = resolved.length

        # Per-sample products of the orthonormalized residuals, centred: their
        # sum is the cross-product matrix whose singular values are the
        # canonical correlations.
        Q_o = _orthonormal_columns(resid_o)
        design = np.ones((n, 1)) if conditioning is None else np.column_stack([np.ones(n), as2d(conditioning)])
        Q_z = _orthonormal_columns(design)
        leverage = np.sum(Q_z ** 2, axis=1)
        hc2 = 1.0 / np.clip(1.0 - leverage, 1e-12, None)
        products = {}
        for nm in names:
            Q_c = _orthonormal_columns(resid_c[nm])
            X = (Q_c[:, :, None] * Q_o[:, None, :]).reshape(n, -1)
            products[nm] = ((X - X.mean(axis=0)) * hc2[:, None], Q_c.shape[1], Q_o.shape[1])

        null = np.zeros((n_resamples, len(names)))
        for start in range(0, n_resamples, MULTIPLIER_CHUNK):
            stop = min(start + MULTIPLIER_CHUNK, n_resamples)
            weights = dependent_wild_weights(time_positions, block, stop - start, rng, independent="rademacher")
            for j, nm in enumerate(names):
                X, p, q = products[nm]
                if p == 0 or q == 0:
                    continue
                null[start:stop, j] = _set_correlation((weights @ X).reshape(-1, p, q))
        return JointTestResult(names, observed, null, diagnostics, block_length=block,
                               dependence=resolved.to_record())

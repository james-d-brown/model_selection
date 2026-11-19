"""
Linear tier: linear-regression residual test (partial canonical correlation).

Z is removed from the candidate and the observations by least squares. The
test statistic is Pillai's trace PT (Pillai, 1955), the sum of the residuals'
squared canonical correlations; the effect size is one minus Wilks' lambda
(Hotelling, 1936), less its chance level. Both are the squared partial
correlation for scalars. Pillai's trace is the squared Frobenius norm of the
cross-product of the orthonormalized residuals, so it is defined for every
null draw, whose reweighted cross-product is not a matrix of correlations
and can have singular values above one, where Wilks' lambda is undefined. It
is the generalised covariance measure's statistic (Lundborg et al., 2022)
for these residuals.

Under H0 every product of a candidate residual and an observation residual
has zero conditional mean, so the null distribution is built by a
multiplier (wild) bootstrap of those products, as in the nonlinear tier: each draw
reweights the per-sample products by multipliers of mean zero and variance
one and recomputes the statistic, without moving either series. The
multipliers are standard normal, as in the nonlinear tier. Independent units
(clusters of events) instead use the wild cluster restricted bootstrap: one
Rademacher sign per unit on the observations' residuals, scaled by cluster
(CR2), which are re-residualized and orthonormalized again in every draw,
so that each draw is computed as the observed statistic is (Cameron,
Gelbach and Miller, 2008; multipliers module). Unlike a
permutation of residuals, which assumes they are exchangeable, the
multiplier bootstrap reproduces the variance of the products when the
residuals' spread varies with Z or with each other (Wu, 1986; Mammen, 1993),
which permutation does not (Winkler et al., 2014). Samples closer than the
dependence lag receive correlated multipliers (the dependent wild bootstrap,
Shao, 2010); on evenly spaced samples the bandwidth is set from the
persistence of the products of the candidate and observation residuals
(significance.block_length), and each draw is divided by a draw of the
fixed-b variance functional (multipliers.fixed_b_draws). Each
product is divided by sqrt(1 - h_t), with h_t the leverage of sample t in
the regression on Z, because least-squares residuals understate the errors'
variance there by the factor 1 - h_t (the HC2 correction, MacKinnon and
White, 1985; Davidson and Flachaire, 2008). All candidates share each set of
multipliers.

Censored series (censoring at high flow) bend at their threshold, so their
conditional means are not linear in Z even when the uncensored series' are.
For each censored series, its fit-then-censor mean given Z is computed from
the uncensored series by least squares (regression.censored, with a
least-squares fit in place of the kernel one: mean, log squared residuals
for the spread, pooled standardized residuals) and added to the
conditioning set as a regressor, for the candidate and the observations
alike: a generated regressor (Heckman, 1979; Pagan, 1984) whose coefficient
least squares estimates freely, so the location-scale model need only hold
up to that recalibration. The test is otherwise unchanged and keeps its
least-squares algebra. The added columns are functions of Z, so the null
hypothesis is unchanged; they differ by candidate, so the observations'
residuals are computed per candidate.

See docs/methods.md, Linear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.association.canonical import canonical_correlations
from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult
from model_selection.statistics.significance.clusters import cluster_scaled_residuals, effective_clusters
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights, fixed_b_draws
from model_selection.statistics.regression.linear import ols_fit
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block
from model_selection.utils import as2d


class _LeastSquares:
    """In-sample least squares on [1, Z], with the residuals() interface of
    the kernel regressions that regression.censored uses."""

    def __init__(self, Z: np.ndarray | None, n: int):
        design = np.ones((n, 1)) if Z is None or np.asarray(Z).size == 0 else np.column_stack([np.ones(n), as2d(Z)])
        self._Q = _orthonormal_columns(design)

    def residuals(self, target: np.ndarray, extra=None):
        T = as2d(np.asarray(target, dtype=float))
        return T - self._Q @ (self._Q.T @ T), None


def censored_mean_columns(candidate, obs, conditioning) -> np.ndarray | None:
    """The fit-then-censor means given the conditioning set, by least
    squares, of whichever of the candidate and the observations are censored
    (those carrying a censoring record), as columns; None when neither is."""
    from model_selection.preprocessing.censoring import censoring_record
    from model_selection.statistics.regression.censored import censored_mean
    n = as2d(obs).shape[0]
    cols = []
    for X in (candidate, obs):
        record = censoring_record(X)
        if record is not None:
            cols.append(censored_mean(record[0], record[1], _LeastSquares(conditioning, n)))
    return None if not cols else np.column_stack(cols)


def augmented_conditioning(candidate, obs, conditioning):
    """The conditioning set with the censored means of the candidate and the
    observations added (censored_mean_columns); unchanged without censoring."""
    extra = censored_mean_columns(candidate, obs, conditioning)
    if extra is None:
        return conditioning
    return extra if conditioning is None or np.asarray(conditioning).size == 0 else \
        np.column_stack([as2d(conditioning), extra])


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


def pillai_trace(resid_c: np.ndarray, resid_o: np.ndarray) -> float:
    """Sum of the squared canonical correlations of two residual matrices
    (already centred)."""
    return float(np.sum(canonical_correlations(resid_c, resid_o) ** 2))


def _trace_of_draws(M: np.ndarray) -> np.ndarray:
    """Pillai's trace of each null draw: the squared Frobenius norm of its
    cross-product of orthonormalized residuals, M (draws, p * q) flattened."""
    return np.sum(M ** 2, axis=-1)


def chance_level(null_draws: np.ndarray, k_y: int, k_x: int) -> float:
    """Mean of 1 - Wilks' lambda for the number of independent samples whose
    null mean of Pillai's trace matches the candidate's own null draws.

    For n independent samples the null mean of Pillai's trace is
    k_y k_x / (n - 1 - k), so the matched n satisfies
    n - 1 - k = k_y k_x / mean(null). Bartlett's (1938) approximation,
    (n - 1 - k - (k_y + k_x + 1) / 2)(-ln Lambda) ~ chi^2(k_y k_x), then gives
    the mean of 1 - Lambda as 1 - exp(-k_y k_x / (n - 1 - k - (k_y + k_x + 1) / 2)),
    in which k cancels (docs/methods.md, Effective sample size). 1 when the null is too wide for the
    approximation (no samples left), 0 when the null is degenerate at zero."""
    mean = float(np.mean(null_draws)) if np.size(null_draws) else 0.0
    if not np.isfinite(mean) or mean <= 0.0:
        return 0.0
    u = float(k_y * k_x)
    remaining = u / mean - (k_y + k_x + 1) / 2.0
    if remaining <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-u / remaining))


MULTIPLIER_CHUNK = 64  # draws generated at a time, to bound memory on long records


class LinearResidualTest(ConditionalIndependenceTest):
    """Pillai's trace of the residuals with a multiplier-bootstrap null; the
    effect size is the partial canonical correlation less its chance level.

    run_length : consecutive insignificant autocorrelations that end the
        decorrelation-lag search (docs/methods.md, Dependence between samples).
    """
    name = "linear"

    def __init__(self, run_length: int = DEFAULT_RUN_LENGTH):
        self.run_length = run_length

    def unrestricted_width_draws(self, candidate, obs, conditioning, n_resamples, rng, time_positions,
                                 dependence_lag, grid_step=None, test: JointTestResult | None = None) -> np.ndarray:
        """Draws of Pillai's trace for the width of the null, for the power
        calculation (statistics.power.effective_sample_size), built as run's
        null except that the observations' residuals are the unrestricted
        ones, given the conditioning set AND the candidate.

        The restricted residuals, given the conditioning set only, contain the
        candidate's effect; resampled, they carry part of it into every draw,
        so a strong effect widens the null and understates the effective
        sample size. The unrestricted residuals estimate the errors whether or
        not the candidate has an effect (Davidson and Flachaire, 2008, compare
        the two kinds of residuals in the wild bootstrap); under the null they
        differ from the restricted ones only by the fitted effect's sampling
        error. The test's p-value keeps the restricted residuals, which impose
        the null.

        On independent units (clusters of events): run's wild cluster
        restricted bootstrap with the unrestricted residuals, scaled by cluster
        (CR2) for the unrestricted fit, re-residualized on the conditioning set
        in each draw. Otherwise: run's multipliers on the per-sample products
        of the orthonormalized candidate residuals and orthonormalized
        unrestricted residuals (whose sum is exactly zero), each divided by
        sqrt(1 - h_t) for the unrestricted fit's leverage, and on a regular
        grid the draws corrected for the bandwidth's bias as
        JointTestResult.width_draws does (undivided draws over the mean of the
        fixed-b functional).

        test : the result of the test whose power is described. Its
            bandwidth and fixed-b mean are then used, so that the effective
            sample size and the degrees of freedom
            (JointTestResult.null_degrees_of_freedom) describe one null.
            Without it the bandwidth is resolved for the candidate's own
            products."""
        n = as2d(obs).shape[0]
        c = as2d(candidate)
        z = augmented_conditioning(candidate, obs, conditioning)
        design = np.ones((n, 1)) if z is None else np.column_stack([np.ones(n), as2d(z)])
        full = np.column_stack([design, c])
        r_c = ols_fit(c, z)[1]
        _, u = ols_fit(as2d(obs), np.column_stack([as2d(z), c]) if z is not None else c)
        shared = test is not None and not (test.dependence or {}).get("independent_units")
        resolved = None if shared else resolve_block(time_positions, dependence_lag, [(r_c, u)], self.run_length,
                                                     grid_step)
        length = test.block_length if shared else resolved.length
        Q_c = _orthonormal_columns(r_c)
        if Q_c.shape[1] == 0:
            return np.zeros(n_resamples)
        if not shared and resolved.independent_units:
            D = _orthonormal_columns(design)
            M = np.eye(n) - D @ D.T
            F = _orthonormal_columns(full)
            M_full = np.eye(n) - F @ F.T
            scaled = cluster_scaled_residuals(as2d(u), M_full, time_positions)
            draws = np.empty(n_resamples)
            for start in range(0, n_resamples, MULTIPLIER_CHUNK):
                stop = min(start + MULTIPLIER_CHUNK, n_resamples)
                weights = dependent_wild_weights(time_positions, resolved.length, stop - start, rng,
                                                 independent="rademacher")
                for b, w in zip(range(start, stop), weights):
                    Q_b = _orthonormal_columns(M @ (scaled * w[:, None]))
                    draws[b] = float(np.sum((Q_c.T @ Q_b) ** 2))
            return draws
        Q_u = _orthonormal_columns(u)
        if Q_u.shape[1] == 0:
            return np.zeros(n_resamples)
        leverage = np.sum(_orthonormal_columns(full) ** 2, axis=1)
        hc2 = 1.0 / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        X = (Q_c[:, :, None] * Q_u[:, None, :]).reshape(n, -1)
        X = (X - X.mean(axis=0)) * hc2[:, None]
        draws = np.empty(n_resamples)
        for start in range(0, n_resamples, MULTIPLIER_CHUNK):
            stop = min(start + MULTIPLIER_CHUNK, n_resamples)
            weights = dependent_wild_weights(time_positions, length, stop - start, rng)
            draws[start:stop] = _trace_of_draws(weights @ X)
        if shared:
            if test.fixed_b is not None:
                draws = draws / float(np.mean(test.fixed_b))
        elif resolved.bandwidth_fraction is not None:
            q = fixed_b_draws(resolved.bandwidth_fraction, n_resamples, rng, n_steps=resolved.grid_length)
            draws = draws / float(np.mean(q))
        return draws

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None,
            period_starts=None):
        names = list(candidates)
        n = as2d(obs).shape[0]
        # Each candidate's conditioning set: with censoring, the censored means
        # of the candidate and the observations are added (module docstring).
        cond = {nm: augmented_conditioning(candidates[nm], obs, conditioning) for nm in names}
        tss_o = float(np.sum((as2d(obs) - as2d(obs).mean(0)) ** 2))
        resid_o, resid_c, hc2, diagnostics = {}, {}, {}, {}
        for name in names:
            z = cond[name]
            _, resid_o[name] = ols_fit(obs, z)
            c = as2d(candidates[name])
            _, r = ols_fit(c, z)
            resid_c[name] = r
            k = 0 if z is None else as2d(z).shape[1]
            design = np.ones((n, 1)) if z is None else np.column_stack([np.ones(n), as2d(z)])
            leverage = np.sum(_orthonormal_columns(design) ** 2, axis=1)
            hc2[name] = 1.0 / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
            tss_c = float(np.sum((c - c.mean(0)) ** 2))
            diagnostics[name] = {
                "r2_candidate_on_selected": 1.0 - float(np.sum(r ** 2)) / tss_c if tss_c > 0 else 0.0,
                "r2_obs_on_selected": 1.0 - float(np.sum(resid_o[name] ** 2)) / tss_o if tss_o > 0 else 0.0,
                "samples_per_conditioning_dimension": n / max(k, 1),
            }
            if z is not conditioning:
                diagnostics[name]["censored_mean_columns"] = k - (0 if conditioning is None else
                                                                  as2d(conditioning).shape[1])
        for nm in names:
            diagnostics[nm]["partial_canonical_correlation"] = partial_canonical_correlation(resid_c[nm], resid_o[nm])
        observed = np.array([pillai_trace(resid_c[nm], resid_o[nm]) for nm in names])
        resolved = resolve_block(time_positions, dependence_lag, [(resid_c[nm], resid_o[nm]) for nm in names],
                                 self.run_length, grid_step)
        block = resolved.length

        units = resolved.independent_units
        null = np.zeros((n_resamples, len(names)))
        if units:
            for nm in names:
                # The effective number of clusters, from the candidate's residuals
                # (significance.clusters.effective_clusters), reported as a diagnostic.
                diagnostics[nm]["effective_clusters"] = effective_clusters(resid_c[nm], time_positions)
            # Independent units (clusters of events): the wild cluster restricted
            # bootstrap (Cameron, Gelbach and Miller, 2008). Each draw multiplies
            # the observations' residuals given the conditioning set, each
            # cluster's scaled by the CR2 correction of that least-squares fit
            # (significance.clusters.cluster_scaled_residuals), by one Rademacher
            # sign per unit, removes the conditioning set from them again, and
            # recomputes Pillai's trace with their own orthonormalization, as the
            # observed statistic is computed. Signs applied to the products
            # instead keep the observed residuals' orthonormalization, which
            # rejects too often with several features and few clusters
            # (multipliers module).
            bases = {}
            for nm in names:
                z = cond[nm]
                design = np.ones((n, 1)) if z is None else np.column_stack([np.ones(n), as2d(z)])
                D = _orthonormal_columns(design)
                bases[nm] = (D, _orthonormal_columns(resid_c[nm]),
                             cluster_scaled_residuals(as2d(resid_o[nm]), np.eye(n) - D @ D.T, time_positions))
            for start in range(0, n_resamples, MULTIPLIER_CHUNK):
                stop = min(start + MULTIPLIER_CHUNK, n_resamples)
                weights = dependent_wild_weights(time_positions, block, stop - start, rng, independent="rademacher")
                for j, nm in enumerate(names):
                    D, Q_c, scaled = bases[nm]
                    if Q_c.shape[1] == 0:
                        continue
                    for b, w in zip(range(start, stop), weights):
                        R = scaled * w[:, None]
                        R = R - D @ (D.T @ R)
                        Q_b = _orthonormal_columns(R)
                        null[b, j] = float(np.sum((Q_c.T @ Q_b) ** 2))
        else:
            # Per-sample products of the orthonormalized residuals, centred: their
            # sum is the cross-product matrix whose singular values are the
            # canonical correlations.
            products = {}
            for nm in names:
                Q_o = _orthonormal_columns(resid_o[nm])
                Q_c = _orthonormal_columns(resid_c[nm])
                X = (Q_c[:, :, None] * Q_o[:, None, :]).reshape(n, -1)
                products[nm] = (X - X.mean(axis=0)) * hc2[nm][:, None]
            for start in range(0, n_resamples, MULTIPLIER_CHUNK):
                stop = min(start + MULTIPLIER_CHUNK, n_resamples)
                weights = dependent_wild_weights(time_positions, block, stop - start, rng)
                for j, nm in enumerate(names):
                    X = products[nm]
                    if X.shape[1] == 0:
                        continue
                    null[start:stop, j] = _trace_of_draws(weights @ X)
        # The fixed-b reference distribution: each draw divided by a draw of the
        # variance functional at the multipliers' bandwidth (fixed_b_draws); all
        # candidates share it, as they share the multipliers.
        q = fixed_b_draws(resolved.bandwidth_fraction, n_resamples, rng, n_steps=resolved.grid_length)
        null /= q[:, None]
        return JointTestResult(names, observed, null, diagnostics, block_length=block,
                               dependence=resolved.to_record(),
                               fixed_b=None if resolved.bandwidth_fraction is None else q)

    def magnitude(self, candidate, obs, conditioning, time_positions, dependence_lag, observed, null_draws,
                  grid_step=None, period_starts=None):
        """1 - Wilks' lambda of the residuals, less its chance level for the
        number of independent samples whose null matches the candidate's own
        null draws of Pillai's trace (chance_level). The null draws are not
        values of 1 - Wilks' lambda, so their mean is not its chance level."""
        z = augmented_conditioning(candidate, obs, conditioning)
        _, resid_o = ols_fit(obs, z)
        _, resid_c = ols_fit(as2d(candidate), z)
        pc = partial_canonical_correlation(resid_c, resid_o)
        chance = chance_level(null_draws, _orthonormal_columns(resid_o).shape[1],
                              _orthonormal_columns(resid_c).shape[1])
        return pc - chance, chance

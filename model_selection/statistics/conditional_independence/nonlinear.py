"""
Nonlinear tier: nonlinear-regression residual test.

Each variable's kernel feature map (its standardized columns plus a radial
basis function kernel, regression.kernel_residuals.feature_kernel) is
regressed on Z by kernel ridge regression with cross-fitting, and the
residual cross-covariance operator is measured by its squared
Hilbert-Schmidt norm, normalised as an RV coefficient (Robert and Escoufier,
1976). Every quantity is computed from the residual feature maps' inner
products with the exact kernels. The construction follows the generalised
Hilbertian covariance measure (Lundborg et al., 2022) and its kernel
extension (Bergen et al., 2026). Under H0 every residual product has zero
conditional mean, so a multiplier bootstrap of the products gives the null
distribution without resampling the candidate; the multipliers follow the
dependent wild bootstrap (Shao, 2010). All candidates share each set of
multipliers. On samples on a regular time grid the multipliers' bandwidth is
set from the persistence of the statistic's own per-sample terms, the
residual feature-map products, from the trace of their autocovariance
matrices (significance.block_length), and each draw is divided by a draw of
the fixed-b variance functional (multipliers.fixed_b_draws). Independent
units (clusters of events) use the wild cluster restricted bootstrap
instead: the observations' feature map is rebuilt in every draw as its
fitted part plus its residual features with one Rademacher sign per unit,
and residualized again with the observed fit's bandwidth and penalty
(_cluster_restricted_null).

With censoring at high flow, a censored variable's conditional means (of its
features and of the variable itself) are estimated by fitting its uncensored
values and then applying the censoring (regression.censored), not by
regressing the censored values, whose conditional means bend at the
threshold. The variable is recognized by its record
(preprocessing.censoring.censoring_record).

See docs/methods.md, Nonlinear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult
from model_selection.preprocessing.censoring import censoring_record
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights, fixed_b_draws
from model_selection.statistics.regression.censored import censored_feature_means
from model_selection.statistics.regression.kernel_residuals import (
    ConditioningRegression, cached_regression, feature_factor_with_cap,
    low_rank,
    shared_regression, spectrum_tasks, standardize,
)
from model_selection.statistics.significance.block_length import (
    DEFAULT_RUN_LENGTH, resolve_block_from_inner_products,
)
from model_selection.utils import as2d

MULTIPLIER_CHUNK = 64  # bootstrap draws generated at a time, to bound memory on long records


class NonlinearResidualTest(ConditionalIndependenceTest):
    """Residual cross-covariance of kernel feature maps after kernel ridge
    removal of Z, calibrated by a multiplier bootstrap.

    run_length : decorrelation-lag run length for the multiplier bandwidth
        (docs/methods.md, Dependence between samples).
    """
    name = "nonlinear"
    uncensored_feature_means = "smear"
    """How the feature means of an uncensored one-column variable are
    estimated given Z. "smear" (the default): derived from the variable's own
    conditional distribution, its regression mean with the location-scale
    smearing of its residuals, as for a censored variable without the
    threshold (regression.censored; docs/methods.md). Each feature is then a
    known function of one estimated distribution, instead of a separate
    regression whose errors add up across the many feature columns.
    "regression": each feature regressed on Z. Variables with several columns
    always use "regression". The smear is the method for a scalar time series
    (the pooled pathway); the event pathway sets "regression" for every event
    sample, whatever its number of components (selection.tiers.build_tier)."""
    def __init__(self, run_length: int = DEFAULT_RUN_LENGTH):
        self.run_length = run_length
        self._factors: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._censored: dict[tuple[int, int], tuple] = {}
        self._uncensored: dict[tuple[int, int], tuple] = {}
        """Uncensored scalar variables' feature means (location-scale smearing),
        by ids of the variable and the regression: the significance test and
        the effect size, and the rounds and comparisons, recur on the same
        pairs. A cached entry is the computation it replaces (it uses no
        random numbers)."""
        self._paths: dict[int, tuple] = {}
        """How each censored variable's conditional means were estimated, with
        the scale and pool diagnostics, by id of the variable."""
        self._own_means: dict[tuple[int, int], tuple] = {}
        """Censored observations' smeared means, by ids of the observations
        and the regression (the reduced model recurs for every candidate)."""
        """Censored variables' feature means, by ids of the variable and the
        regression (both recur across the rounds and comparisons of a trial)."""
        self.magnitude_diagnostics: dict[int, dict] = {}
        """Regression diagnostics of each effect size, by id of the candidate
        array: whether a cross-fitting record was short, and the largest share
        of a kernel's trace left out by a factor at the rank cap."""

    def _factor(self, X: np.ndarray) -> np.ndarray:
        """Kernel factor of a variable, cached by array identity: the same
        arrays recur in every round of a selection trial."""
        cached = self._factors.get(id(X))
        if cached is None or cached[0] is not X:
            cached = (X, *self._make_factor(X))
            self._factors[id(X)] = cached
        return cached[1]

    def _make_factor(self, X: np.ndarray) -> tuple:
        """(factor, cap share)."""
        return feature_factor_with_cap(X)

    def _factor_cap(self, X: np.ndarray) -> float:
        cached = self._factors.get(id(X))
        return cached[2] if cached is not None and cached[0] is X else 0.0

    def _prepare(self, variables: list, conditioning, time_positions, dependence_lag, period_starts=None) -> None:
        """Compute, in one parallel pool, every eigendecomposition the round
        needs that is not yet kept: the conditioning regression's, one per
        trial bandwidth, and each variable's feature factor. They are
        independent and use no random numbers, so results do not depend on
        the number of workers."""
        from model_selection.parallel import run_tasks
        tasks, slots = [], []
        if conditioning is not None and np.asarray(conditioning).size \
                and cached_regression(conditioning, time_positions, dependence_lag, period_starts) is None:
            spectra = spectrum_tasks(conditioning)
            tasks += spectra
            slots += [("regression", None)] * len(spectra)
        for X in variables:
            cached = self._factors.get(id(X))
            if cached is None or cached[0] is not X:
                tasks.append(lambda X=X: self._make_factor(X))
                slots.append(("factor", X))
        if not tasks:
            return
        results = run_tasks(tasks)
        spectra = [r for (kind, _), r in zip(slots, results) if kind == "regression"]
        if spectra:
            shared_regression(conditioning, time_positions, buffer=dependence_lag, spectra=spectra,
                              period_starts=period_starts)
        for (kind, X), r in zip(slots, results):
            if kind == "factor":
                self._factors[id(X)] = (X, *r)

    def _residuals(self, X: np.ndarray, regression: ConditioningRegression | None):
        """(centred residual coordinates of the feature map, residuals of the
        variable's own standardized columns, cross-fitted R^2 of those columns)."""
        L = self._factor(X)
        own = standardize(X)
        if regression is None:
            return L - L.mean(axis=0), own, 0.0
        record = censoring_record(X)
        if record is None and self.uncensored_feature_means != "regression" and as2d(X).shape[1] == 1:
            key = (id(X), id(regression))
            cached = self._uncensored.get(key)
            if cached is None or cached[0] is not X or cached[1] is not regression or cached[2] is not L:
                means, mean, _ = censored_feature_means(X, L, as2d(X)[:, 0], None, regression)
                cached = (X, regression, L, means, mean)
                self._uncensored[key] = cached
            means, mean = cached[3], cached[4]
            E, E_own = L - means, _own_residuals(X, mean)
        elif record is None:
            E, E_own = regression.residuals(L, own)
        else:
            key = (id(X), id(regression))
            cached = self._censored.get(key)
            if cached is None or cached[0] is not X or cached[1] is not regression:
                uncensored, threshold = record
                cached = (X, regression, *censored_feature_means(X, L, uncensored, threshold, regression))
                self._paths[id(X)] = (X, {"censored_means": "fit_then_censor", **cached[4]})
                self._censored[key] = cached
            means, mean = cached[2], cached[3]
            E, E_own = L - means, _own_residuals(X, mean)
        tss = float(np.sum(own ** 2))
        r2_own = 1.0 - float(np.sum(E_own ** 2)) / tss if tss > 0 else 0.0
        return E - E.mean(axis=0), E_own - E_own.mean(axis=0), r2_own

    def censored_path(self, X: np.ndarray, regression) -> dict:
        """How X's conditional means were estimated: {} for an uncensored
        variable; for a censored one, "fit_then_censor" with the scale and
        pool diagnostics, or "regression" where there was nothing to condition
        on (its conditional mean is then its mean)."""
        if censoring_record(X) is None:
            return {}
        if regression is None:
            return {"censored_means": "regression"}
        entry = self._paths.get(id(X))
        return dict(entry[1]) if entry is not None and entry[0] is X else {"censored_means": "fit_then_censor"}

    def _regression(self, conditioning, time_positions, dependence_lag,
                    period_starts=None) -> ConditioningRegression | None:
        """Cross-fitted regression on the conditioning set, reused for the
        same arrays (the significance test and the magnitude of its winner)."""
        if conditioning is None or np.asarray(conditioning).size == 0:
            return None
        return shared_regression(conditioning, time_positions, buffer=dependence_lag, period_starts=period_starts)

    def _feature_direction(self, candidate, obs, conditioning, time_positions, dependence_lag,
                           period_starts=None) -> dict:
        """The effect size's parts (magnitude): the held-out error reduction
        of the observations' residuals on Z by the feature direction, at the
        factor's rank and at half of it, the correlation form for one
        observation column, and diagnostics."""
        from model_selection.statistics.regression.kernel_residuals import (
            BANDWIDTH_MULTIPLIERS as MULTS, _median_bandwidth, _rbf_column, crossfit_blocks, _folds as make_folds,
            low_rank, pivoted_cholesky, standardize as kstd,
        )
        regression = self._regression(conditioning, time_positions, dependence_lag, period_starts)
        E_c, own_c, _ = self._residuals(candidate, regression)
        _, own_o, _ = self._residuals(obs, regression)
        e = own_o - own_o.mean(axis=0)
        D = [own_c[:, j] for j in range(own_c.shape[1])]
        record = censoring_record(candidate)
        if record is not None and regression is not None:
            from model_selection.statistics.regression.censored import location_scale, reduced_pool
            U, t = record
            mu, scale, pool, _ = location_scale(U, regression)
            values, weights = reduced_pool(pool)
            p_exceed = (mu[:, None] + scale[:, None] * values[None, :] > t) @ weights
            D.append((np.asarray(U) > t).astype(float) - p_exceed)
        D = [d / np.sqrt(np.mean(d ** 2)) if np.mean(d ** 2) > 0 else d for d in D]
        F = np.column_stack(D + [E_c])
        n = F.shape[0]
        cap = low_rank() or n
        cap_share = 0.0
        if regression is None:
            block, k, short = crossfit_blocks(time_positions, dependence_lag, period_starts)
            folds, kept = make_folds(block, k, 1)
            designs, blocks_short = [F], bool(short or kept)
        else:
            folds, blocks_short = regression._folds, regression.blocks_short
            Zs = kstd(as2d(np.asarray(conditioning, dtype=float)))
            base, sq = _median_bandwidth(Zs), (Zs ** 2).sum(1)
            diag = 2.0 * (F ** 2).sum(1)
            designs = []
            for m in MULTS:
                rbf = _rbf_column(Zs, sq, base * m)
                G, share = pivoted_cholesky(diag.copy(), lambda p, rbf=rbf: (1.0 + rbf(p)) * (F @ F[p]), cap,
                                            kind="effect")
                designs.append(G)
                cap_share = max(cap_share, share)
        # Several observation columns are whitened, so that the penalty and
        # bandwidth are chosen for every column alike; the determinant ratio
        # does not depend on the whitening.
        w, V = np.linalg.eigh(e.T @ e / n)
        keep = w > max(float(w.max()), 0.0) * e.shape[1] * np.finfo(float).eps
        ew = e @ (V[:, keep] / np.sqrt(w[keep]))
        nested = True
        if ew.shape[1] == 0:
            effect = half = corr = 0.0
        else:
            chosen: list = []
            h, nested = _ridge_direction(ew, designs, folds, time_positions, dependence_lag, chosen)
            effect = _error_reduction(ew, h)
            # The refit at half the rank feeds only the rank-cap flag, so it is
            # made only when a factor reached the cap (low_rank_material). It
            # keeps each block's bandwidth and penalty, so that it measures the
            # rank alone, not a second choice of tuning.
            half = effect
            if cap_share > 0:
                h_half = _ridge_refit(ew, [G[:, :max(1, G.shape[1] // 2)] for G in designs], folds, chosen)
                half = _error_reduction(ew, h_half)
            corr = _held_out_r2(ew[:, 0], h[:, 0], folds) if ew.shape[1] == 1 else float("nan")
        return {"features": effect, "features_half": half, "features_corr": corr,
                "diagnostics": {"crossfit_blocks_short": bool(blocks_short),
                                "low_rank_cap_share": max(cap_share, 0.0 if regression is None
                                                          else regression.cap_share),
                                "low_rank_rank_change": 0.0 if regression is None else regression.rank_change,
                                "effect_half_rank_change": abs(effect - half),
                                "effect_tuning_nested": bool(nested)}}

    def magnitude(self, candidate, obs, conditioning, time_positions, dependence_lag, observed, null_draws,
                  grid_step=None, full: ConditioningRegression | None = None, period_starts=None):
        """The candidate's share of the variation in the observations left
        unexplained by Z, as the held-out error reduction of a direction
        built from the candidate and Z only (docs/methods.md, Nonlinear tier).

        F holds the candidate's own residuals given Z and, when censored, its
        exceedance-indicator residual (each scaled to unit root mean square),
        then its residual feature coordinates (the test's R_phi). The
        direction h is ridge regression of e, the observations' residuals on
        Z, on the factor of the kernel (1 + k_Z) * (F F'), k_Z the tier's
        radial basis function kernel on the standardized Z at each of its
        bandwidths: functions b(Z)'F whose coefficients may be constant or
        vary smoothly with the state (varying-coefficient models; Hastie and
        Tibshirani, 1993), and, through the features, non-monotone in the
        candidate. The factor is pivoted Cholesky to the tier's rank cap; the
        bandwidth and penalty are chosen by the tier's blocked
        cross-validation, and each block is predicted by a fit without it.
        The effect is
            max(0, 1 - det((e - h)'(e - h)) / det(e'e)),
        for one column 1 - SS(e - h) / SS(e). For any h fitted without the
        predicted block, its expectation is at most the share of e's
        variation explained by E[O | Z, C], so it is a lower bound on the
        candidate's share. Every regressor is a function of Z and C only, so
        under the null h's covariance with e is a product of the two
        regressions' errors, as for the test (Shah and Peters, 2020).
        `full` is accepted for compatibility and not used. Returns
        (effect, None)."""
        out = self._feature_direction(candidate, obs, conditioning, time_positions, dependence_lag, period_starts)
        self.magnitude_diagnostics[id(candidate)] = out["diagnostics"]
        return out["features"], None

    def magnitudes(self, candidates: dict, obs, conditioning, time_positions, dependence_lag,
                   period_starts=None) -> dict:
        """magnitude() of several candidates, in parallel."""
        from model_selection.parallel import parallel_map
        names = list(candidates)
        values = parallel_map(lambda nm: self.magnitude(candidates[nm], obs, conditioning, time_positions,
                                                        dependence_lag, 0.0, np.zeros(1),
                                                        period_starts=period_starts), names)
        return dict(zip(names, values))

    def _cluster_restricted_null(self, names, fitted, obs, E_o, regression, weight_chunks) -> tuple[np.ndarray, dict]:
        """Null draws for independent units (clusters of events): the wild
        cluster restricted bootstrap (Cameron, Gelbach and Miller, 2008),
        written out for the tier's cross-fitted kernel ridge regression.

        With A the regression's residual maker at the bandwidth and penalty
        chosen for the observations' feature map L, exactly as for the observed
        residual features, and C the centring, so that E_o = C A L
        (ConditioningRegression.residual_operator; _residuals), each draw
        rebuilds the feature map under the null, L* = (L - A L) + w o (A L),
        its fitted part plus its uncentred residuals with one Rademacher sign
        per unit, and recomputes the residual features E* = C A L* and the RV
        coefficient with each candidate's residual features, as the observed
        statistic is computed. The draw with every sign +1 is the observed
        statistic. C A is linear, so C A (L - A L) is computed once and every
        draw of a chunk is residualized in one pass. In each draw a fold's
        prediction uses only the other folds' rebuilt values, so a unit's sign
        never enters its own prediction; the bandwidth and penalty, fixed
        before any draw, hold no draw's signs. Holding them fixed rather than
        choosing them again in each draw is an approximation, asymptotic (as
        in Hardle and Mammen, 1993; Lin, 1997). No published result covers
        this construction with few clusters; its level there rests on
        simulation (docs/methods.md, Dependence between samples).

        Returns the draws and a record: whether A L reproduced E_o (it does
        on the regression path, which every event sample takes;
        selection.tiers.build_tier), and the share of the fitted part's
        residual, ||C A (L - A L)||^2 / ||E_o||^2, which grows with the smoothing
        bias when Z is informative."""
        L = self._factor(obs)
        if regression is None:
            residualize = lambda Y: Y - Y.mean(axis=0)
        else:
            residualize = regression.residual_operator(L)
        # The observed residual features are the cross-fitted residuals, centred
        # (_residuals): E_o = C A L. The data are rebuilt from the uncentred
        # residuals A L, and each draw centred after its residualization.
        apply = lambda Y: (lambda R: R - R.mean(axis=0))(residualize(Y))
        resid = residualize(L)
        E_o = np.asarray(E_o, dtype=float)
        identity_error = float(np.linalg.norm(apply(L) - E_o) / max(np.linalg.norm(E_o), 1e-300))
        base = apply(L - resid)
        record = {"cluster_draws_reproduce_statistic": bool(identity_error < 1e-8),
                  "cluster_draws_fitted_share": float(np.sum(base ** 2) / max(np.sum(E_o ** 2), 1e-300))}
        n, p = E_o.shape
        E_cs = [fitted[nm][0] for nm in names]
        norms_c = [float(np.linalg.norm(E_c.T @ E_c)) for E_c in E_cs]
        chunks = []
        for _, _, weights in weight_chunks():
            B = weights.shape[0]
            stacked = (weights[:, :, None] * resid[None, :, :]).transpose(1, 0, 2).reshape(n, -1)
            R = apply(stacked).reshape(n, B, p) + base[:, None, :]          # (n, B, p)
            norm_r = np.linalg.norm(np.einsum("nbp,nbq->bpq", R, R), axis=(1, 2))
            out = np.zeros((B, len(names)))
            for j, (E_c, norm_c) in enumerate(zip(E_cs, norms_c)):
                cross = np.einsum("nk,nbp->bkp", E_c, R)
                denom = norm_c * norm_r
                out[:, j] = np.where(denom > 0, np.sum(cross ** 2, axis=(1, 2)) / np.where(denom > 0, denom, 1.0), 0.0)
            chunks.append(out)
        return np.concatenate(chunks, axis=0), record

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None,
            period_starts=None):
        names = list(candidates)
        n = as2d(obs).shape[0]
        self._prepare([obs] + [candidates[nm] for nm in names], conditioning, time_positions, dependence_lag,
                      period_starts)
        regression = self._regression(conditioning, time_positions, dependence_lag, period_starts)
        E_o, own_o, r2_o = self._residuals(obs, regression)
        # In the low-rank path the statistic is computed from the per-sample
        # vectors u_i = e_C,i (x) e_O,i (Kronecker product of the residual
        # coordinates), whose inner products are the entries of A below, so
        # that no n x n matrix is formed; the two forms are equal.
        features = low_rank() is not None and n > low_rank()
        G_o = None if features else E_o @ E_o.T
        norm_o = float(np.linalg.norm(E_o.T @ E_o)) if features else float(np.sqrt(np.sum(G_o ** 2)))
        from model_selection.parallel import parallel_map
        fitted = dict(zip(names, parallel_map(lambda nm: self._residuals(candidates[nm], regression), names)))

        def feature_terms(E_c):
            """(sum of the u_i, centred u_i as rows, normaliser)."""
            F = (E_c[:, :, None] * E_o[:, None, :]).reshape(n, -1)
            total = F.sum(axis=0)
            return total, F - total / n, float(np.linalg.norm(E_c.T @ E_c)) * norm_o

        def centred_products(E_c):
            """Inner products of the per-sample terms of the statistic (the
            residual products, centred at their mean) and the normaliser."""
            G_c = E_c @ E_c.T
            A = G_c * G_o
            row = A.mean(axis=1)
            return A, A - row[:, None] - row[None, :] + A.mean(), float(np.sqrt(np.sum(G_c ** 2))) * norm_o

        # Multiplier bandwidth from the persistence of the statistic's own
        # per-sample terms (all candidates share one set of multipliers).
        resolved = resolve_block_from_inner_products(
            time_positions, dependence_lag,
            [(lambda nm=nm: feature_terms(fitted[nm][0])[1]) if features else centred_products(fitted[nm][0])[1]
             for nm in names],
            self.run_length, grid_step,
            factors=[((lambda nm=nm: fitted[nm][0]), (lambda: E_o)) if features
                     else (fitted[nm][0] @ fitted[nm][0].T, G_o) for nm in names])
        block = resolved.length
        # Independent units (clusters of events) use the wild cluster restricted
        # bootstrap instead of multipliers on the products (below).
        units = resolved.independent_units
        # All candidates share one set of multipliers. They are generated in
        # chunks of draws, and regenerated for each candidate from the same
        # generator state, so that only one chunk and one candidate's terms are
        # held at a time; the draws are those of a single generation of all
        # n_resamples at once.
        start_state = rng.bit_generator.state

        def weight_chunks():
            gen = np.random.Generator(type(rng.bit_generator)())
            gen.bit_generator.state = start_state
            for start in range(0, n_resamples, MULTIPLIER_CHUNK):
                stop = min(start + MULTIPLIER_CHUNK, n_resamples)
                yield start, stop, dependent_wild_weights(time_positions, block, stop - start, gen,
                                                          independent="rademacher" if units else "gaussian")
            rng.bit_generator.state = gen.bit_generator.state

        if not names:                                  # the generator advances as for any draws
            for _ in weight_chunks():
                pass
        observed = np.empty(len(names))
        null = np.empty((n_resamples, len(names)))
        diagnostics = {}
        k = 0 if conditioning is None else as2d(conditioning).shape[1]
        for j, nm in enumerate(names):
            E_c, _, r2_c = fitted[nm]
            if features:
                total, Fc, denom = feature_terms(E_c)
            else:
                A, P, denom = centred_products(E_c)
            for start, stop, weights in ([] if units else weight_chunks()):
                if denom > 0 and features:
                    null[start:stop, j] = np.sum((weights @ Fc) ** 2, axis=1) / denom
                elif denom > 0:
                    null[start:stop, j] = np.einsum("bi,bi->b", weights @ P, weights) / denom
                else:
                    null[start:stop, j] = 0.0
            if denom > 0 and features:
                observed[j] = float(total @ total) / denom
            elif denom > 0:
                observed[j] = float(A.sum()) / denom
            else:
                observed[j] = 0.0
            diagnostics[nm] = {
                "r2_candidate_on_selected": r2_c,
                "r2_obs_on_selected": r2_o,
                "samples_per_conditioning_dimension": n / max(k, 1),
                **self.censored_path(candidates[nm], regression),
            }
        cluster_record = {}
        if units:
            null, cluster_record = self._cluster_restricted_null(names, fitted, obs, E_o, regression, weight_chunks)
        # The fixed-b reference distribution, as in the linear tier.
        q = fixed_b_draws(resolved.bandwidth_fraction, n_resamples, rng, n_steps=resolved.grid_length)
        null /= q[:, None]
        dependence = resolved.to_record()
        dependence["crossfit_blocks_short"] = bool(regression is not None and regression.blocks_short)
        dependence["low_rank_cap_share"] = 0.0 if regression is None else regression.cap_share
        dependence["low_rank_rank_change"] = 0.0 if regression is None else regression.rank_change
        dependence["feature_factor_cap_share"] = max(self._factor_cap(X) for X in [obs] + [candidates[nm] for nm in names])
        dependence.update({f"obs_{k}": v for k, v in self.censored_path(obs, regression).items()})
        dependence.update(cluster_record)
        return JointTestResult(names, observed, null, diagnostics, block_length=block, dependence=dependence,
                               fixed_b=None if resolved.bandwidth_fraction is None else q)


def _own_residuals(X: np.ndarray, conditional_mean: np.ndarray) -> np.ndarray:
    """Residuals of X's own standardized column about a given conditional
    mean of X, on the scale of standardize(X)."""
    x = as2d(np.asarray(X, dtype=float))[:, 0]
    sd = x.std()
    return ((x - conditional_mean) / (sd if sd > 0 else 1.0))[:, None]


MIN_NESTED_BLOCKS = 8
"""Fewest cross-fitting blocks for which the effect size's tuning is nested
(_ridge_direction): with groups of TUNING_WATER_YEARS blocks, each group's
tuning then has at least two other groups."""

TUNING_WATER_YEARS = 2
"""Cross-fitting blocks, each a water year, in each outer fold of the effect
size's nested tuning (_ridge_direction): the smallest number for which the
tuning's cost stays near that of the tier's other computations; a fold of
one water year costs several times as much at the rank cap."""


def _ridge_coefficients(G: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(columns, observation columns, penalties): ridge coefficients for every
    penalty (RIDGE_PENALTIES, relative to the mean eigenvalue of the Gram
    matrix G), from one eigendecomposition, with cross-product c."""
    from model_selection.statistics.regression.kernel_residuals import RIDGE_PENALTIES
    lam = np.asarray(RIDGE_PENALTIES, dtype=float)
    ev, Q = np.linalg.eigh((G + G.T) / 2.0)
    scale = max(float(np.trace(G)) / G.shape[0], 1e-300)
    a = (Q.T @ c)[:, :, None] / (np.clip(ev, 0, None)[:, None, None] + lam[None, None, :] * scale)
    return (Q @ a.reshape(G.shape[0], -1)).reshape(G.shape[0], c.shape[1], lam.size)


def _predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """(rows, observation columns, penalties): predictions of X's rows."""
    return (X @ coef.reshape(X.shape[1], -1)).reshape(X.shape[0], coef.shape[1], coef.shape[2])


def _tuning_groups(folds: list, time_positions: np.ndarray, dependence_lag: float):
    """The outer folds of the effect size's nested tuning: groups of
    TUNING_WATER_YEARS consecutive blocks (a remainder joins the last group),
    each with a buffer of the samples within one dependence lag of its span.
    Returns the groups (block indices), the atomic segments of the record
    (rows), and each group's rows R and buffer B as sets of segments: every
    set the tuning uses is a union of segments, whose Gram matrices are then
    formed once."""
    rows = [p for p, _ in folds]
    K, size = len(rows), TUNING_WATER_YEARS
    starts = list(range(0, K, size))
    if len(starts) > 1 and K - starts[-1] < size:
        starts.pop()
    groups = [list(range(a, starts[j + 1] if j + 1 < len(starts) else K)) for j, a in enumerate(starts)]
    t = np.asarray(time_positions, dtype=float)
    member = np.zeros((t.size, 2 * len(groups)), bool)
    for g, blocks in enumerate(groups):
        r = np.concatenate([rows[b] for b in blocks])
        member[r, 2 * g] = True
        lo, hi = float(t[r].min()), float(t[r].max())
        member[:, 2 * g + 1] = ((t >= lo - dependence_lag) & (t < lo)) | ((t > hi) & (t <= hi + dependence_lag))
        member[r, 2 * g + 1] = False
    signature, segment = np.unique(member, axis=0, return_inverse=True)
    segment = np.asarray(segment).ravel()
    segments = [np.flatnonzero(segment == k) for k in range(len(signature))]
    R = [frozenset(np.flatnonzero(signature[:, 2 * g])) for g in range(len(groups))]
    B = [frozenset(np.flatnonzero(signature[:, 2 * g + 1])) for g in range(len(groups))]
    return groups, segments, R, B


def _ridge_direction(e: np.ndarray, designs: list, folds: list, time_positions: np.ndarray,
                     dependence_lag: float, choices: list | None = None) -> tuple[np.ndarray, bool]:
    """(held-out predictions of e, whether the tuning was nested): ridge
    regression of e on one of `designs` (one per bandwidth), each block
    predicted by a fit on its fold's fitting samples.

    The design and penalty used for a block are chosen without that block
    (nested cross-validation; Stone, 1974; Varma and Simon, 2006): the blocks
    are grouped in TUNING_WATER_YEARS consecutive water years, and those of
    group g are chosen by cross-validation over the other groups, each
    predicted, outside g's buffer, by a fit that leaves out both groups and
    a buffer of one dependence lag on each side of each (_tuning_groups).
    Choosing them by the held-out errors of the predicted blocks themselves
    would make the reported error the smallest of many noisy estimates,
    which is optimistic. With fewer than MIN_NESTED_BLOCKS blocks the design
    and penalty are chosen by all blocks' held-out errors, and the second
    value is False.

    Each block's Gram matrix, and each segment's, is formed once; a fit's
    Gram matrix is the total less those it leaves out, and its
    eigendecomposition gives the coefficients for every penalty, and serves
    both groups whose exclusions it combines.

    choices: when given, a list that receives, for each block, the (design,
    penalty) indices used to predict it (for _ridge_refit)."""
    e2 = e if e.ndim == 2 else e[:, None]
    designs = [X for X in designs if X.shape[1] > 0]
    if not designs:
        return np.zeros_like(e), False
    rows = [p for p, _ in folds]
    n, K = e2.shape[0], len(rows)
    label = np.full(n, -1)
    for k, r in enumerate(rows):
        label[r] = k
    left_out = []
    for _, fit in folds:
        keep = np.zeros(n, bool)
        keep[fit] = True
        left_out.append([int(m) for m in np.unique(label[~keep])])
    nested = K >= MIN_NESTED_BLOCKS
    if nested:
        groups, segments, R, B = _tuning_groups(folds, time_positions, dependence_lag)
    full, err = [], None
    for d, X in enumerate(designs):
        Gb = [X[r].T @ X[r] for r in rows]
        cb = [X[r].T @ e2[r] for r in rows]
        Gt, ct = sum(Gb), sum(cb)
        # Each block's predictions, for every penalty, from its usual fit.
        full.append([_predict(X[rows[i]], _ridge_coefficients(Gt - sum((Gb[m] for m in left_out[i]),
                                                                         np.zeros_like(Gt)),
                                                              ct - sum((cb[m] for m in left_out[i]),
                                                                       np.zeros_like(ct))))
                     for i in range(K)])
        if not nested:
            continue
        if err is None:
            err = np.zeros((len(groups), len(designs), full[0][0].shape[2]))
        Gs = [X[r].T @ X[r] for r in segments]
        cs = [X[r].T @ e2[r] for r in segments]
        for g in range(len(groups)):
            for h in range(g + 1, len(groups)):
                serve_g, serve_h = R[h] - B[g], R[g] - B[h]   # h's rows outside g's buffer tune g
                if not serve_g and not serve_h:
                    continue
                excl = R[g] | B[g] | R[h] | B[h]
                coef = _ridge_coefficients(Gt - sum((Gs[k] for k in excl), np.zeros_like(Gt)),
                                           ct - sum((cs[k] for k in excl), np.zeros_like(ct)))
                for target, segs in ((g, serve_g), (h, serve_h)):
                    if segs:
                        r = np.concatenate([segments[k] for k in segs])
                        err[target, d] += ((e2[r][:, :, None] - _predict(X[r], coef)) ** 2).sum((0, 1))
    out = np.zeros_like(e2)
    if not nested:
        total = np.array([[sum(((e2[rows[i]][:, :, None] - full[d][i]) ** 2).sum((0, 1)) for i in range(K))]
                          for d in range(len(designs))])[:, 0, :]
        d, j = np.unravel_index(int(np.argmin(total)), total.shape)
        chosen = [(int(d), int(j))] * K
    else:
        chosen = [None] * K
        for g, blocks in enumerate(groups):
            d, j = np.unravel_index(int(np.argmin(err[g])), err[g].shape)
            for i in blocks:
                chosen[i] = (int(d), int(j))
    for i, (d, j) in enumerate(chosen):
        out[rows[i]] = full[d][i][:, :, j]
    if choices is not None:
        choices.extend(chosen)
    return (out if e.ndim == 2 else out[:, 0]), nested


def _ridge_refit(e: np.ndarray, designs: list, folds: list, choices: list) -> np.ndarray:
    """Held-out predictions of e by ridge regression with the design and
    penalty chosen for each block by _ridge_direction (choices), on other
    designs of the same order, such as the same factors at half the rank:
    each block predicted by a fit on its fold's fitting samples, with no
    tuning. The refit then differs from the original only through the
    designs."""
    e2 = e if e.ndim == 2 else e[:, None]
    designs = [X for X in designs if X.shape[1] > 0]
    out = np.zeros_like(e2)
    if not designs:
        return out if e.ndim == 2 else out[:, 0]
    rows = [p for p, _ in folds]
    n, K = e2.shape[0], len(rows)
    label = np.full(n, -1)
    for k, r in enumerate(rows):
        label[r] = k
    left_out = []
    for _, fit in folds:
        keep = np.zeros(n, bool)
        keep[fit] = True
        left_out.append([int(m) for m in np.unique(label[~keep])])
    for d in sorted({c[0] for c in choices}):
        X = designs[d]
        Gb = [X[r].T @ X[r] for r in rows]
        cb = [X[r].T @ e2[r] for r in rows]
        Gt, ct = sum(Gb), sum(cb)
        for i in (i for i in range(K) if choices[i][0] == d):
            coef = _ridge_coefficients(Gt - sum((Gb[m] for m in left_out[i]), np.zeros_like(Gt)),
                                       ct - sum((cb[m] for m in left_out[i]), np.zeros_like(ct)))
            out[rows[i]] = _predict(X[rows[i]], coef)[:, :, choices[i][1]]
    return out if e.ndim == 2 else out[:, 0]


def _error_reduction(e: np.ndarray, h: np.ndarray) -> float:
    """max(0, 1 - det((e - h)'(e - h)) / det(e'e)) for residual columns e
    and their held-out predictions h."""
    e, h = as2d(e), as2d(h)
    sign_r, logdet_r = np.linalg.slogdet(e.T @ e)
    sign_f, logdet_f = np.linalg.slogdet((e - h).T @ (e - h))
    if sign_r <= 0 or sign_f <= 0:
        return 0.0
    return max(0.0, float(1.0 - np.exp(logdet_f - logdet_r)))


def _held_out_r2(e: np.ndarray, h: np.ndarray, folds: list) -> float:
    """max(S^2 - V, 0) / (sum h^2 sum e^2), 0 if S <= 0: S = sum e h, V the
    variance estimate of S from its sums over the blocks."""
    S = float(e @ h)
    K = len(folds)
    if S <= 0 or K < 2:
        return 0.0
    Sk = np.array([float(e[p] @ h[p]) for p, _ in folds])
    V = K / (K - 1) * float(np.sum((Sk - S / K) ** 2))
    denom = float(np.sum(h ** 2) * np.sum(e ** 2))
    return max(S ** 2 - V, 0.0) / denom if denom > 0 else 0.0

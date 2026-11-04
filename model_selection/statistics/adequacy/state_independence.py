"""
State-independence (no effect modification) adequacy test for the
nonlinear-regression tier.

That tier measures the candidate-obs dependence averaged over the states of
the conditioning set Z, so a relationship whose direction changes across
states can average out and go undetected. This test asks instead whether the
observations depend on the candidate jointly with Z (an interaction), which
does not average over states. It detects any interaction in the conditional
mean, not only one that changes sign.

The test is a kernel-machine variance-component score test for a
candidate-by-Z interaction (Lin 1997; Maity & Lin 2011; the SKAT framework,
Wu et al. 2011). Only the null model (additive main effects, K_c + K_z) is
fitted, by kernel ridge regression with a REML-estimated penalty; the score
is trace(r' (K_c * K_z) r) on its residuals r. The interaction surface is
never parameterized, so power does not collapse as Z grows.

The null distribution is a wild bootstrap of the null model's residuals,
conditional on the null fit: the penalties are estimated once, on the
observed data, and held fixed in every draw, as kernel-machine score tests
condition on the fitted null model (Lin, 1997; Wu et al., 2011). A kernel
ridge fit uses many degrees of freedom, so its residuals are smaller than the
errors; each residual is divided by its standard deviation under the fit,
the square root of the diagonal of (I - H)(I - H)' with H the smoother
matrix (the covariance of a linear smoother's residuals; Buja, Hastie and
Tibshirani, 1989), which reduces to the HC2 correction for least squares.
Without it the null is too narrow and the test rejects far too often on
multivariate samples.

Beyond the rank cap each kernel (K_z, K_main, K_main + K_int, K_int) is a
pivoted-Cholesky factor, used through its thin spectrum
(regression.kernel_residuals); the statistic, the penalties, the effect sizes
and every bootstrap draw use the same factors, so an approximation at the cap
changes which departures the test is most sensitive to, not its calibration.
Below the cap every quantity is the one computed from the kernels' full
eigendecompositions.

Limitations: samples longer than the shared kernel sample cap
(SelectionConfig.nonlinear_max_n) are subsampled once, evenly, before
testing; and only the conditional mean is tested, not Z-dependent variance.

See docs/methods.md, Nonlinear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.adequacy.base import (
    AdequacyTestFunc,
    FunctionalFormTest,
    FunctionalFormTestResult,
    as_adequacy_test,
)
from model_selection.statistics.adequacy.kernels import (
    kernel_spectrum,
    project,
    rbf_kernel,
    reml_lambda_from_spectrum,
    variance_shares,
)
from model_selection.statistics.regression.kernel_residuals import _median_bandwidth, _rbf_column, _use_low_rank
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH
from model_selection.statistics.significance.resampling import grid_subsample
from model_selection.utils import as2d


def _scaled_inputs(candidate: np.ndarray, conditioning: np.ndarray):
    """(candidate, standardized conditioning set, their median-heuristic
    bandwidths)."""
    Z = as2d(conditioning)
    z_std = Z.std(axis=0, ddof=1)
    z_std_safe = np.where(z_std > 0, z_std, 1.0)
    Z_scaled = (Z - Z.mean(axis=0)) / z_std_safe
    candidate_2d = as2d(candidate)
    return candidate_2d, Z_scaled, _median_bandwidth(candidate_2d), _median_bandwidth(Z_scaled)


def kernel_main_effect_kernels(
        candidate: np.ndarray, conditioning: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RBF kernels over the candidate (K_c) and over the standardized joint
    conditioning set (K_z), each with a median-heuristic bandwidth. Candidate
    may be scalar or multivariate.
    """
    candidate_2d, Z_scaled, bw_c, bw_z = _scaled_inputs(candidate, conditioning)
    return rbf_kernel(candidate_2d, bw_c), rbf_kernel(Z_scaled, bw_z)


def interaction_spectra(candidate: np.ndarray, conditioning: np.ndarray) -> dict:
    """The spectra (eigenvalues, eigenvectors, cap share) of the conditioning
    kernel K_z, the null kernel K_main = K_c + K_z and the augmented kernel
    K_main + K_int, and a function applying the interaction kernel
    K_int = K_c * K_z (Hadamard product). They depend on the candidate and the
    conditioning set only, so one computation serves the observed statistic
    and every resampled one. Within the rank cap they come from the full
    kernels; beyond it from pivoted-Cholesky factors, computed as parallel
    tasks."""
    from model_selection.parallel import run_tasks
    candidate_2d, Z_scaled, bw_c, bw_z = _scaled_inputs(as2d(candidate), conditioning)
    n = candidate_2d.shape[0]
    if not _use_low_rank(n):
        K_c, K_z = rbf_kernel(candidate_2d, bw_c), rbf_kernel(Z_scaled, bw_z)
        K_int = K_c * K_z
        z, main, full = run_tasks([lambda: kernel_spectrum(n, None, None, lambda: K_z),
                                   lambda: kernel_spectrum(n, None, None, lambda: K_c + K_z),
                                   lambda: kernel_spectrum(n, None, None, lambda: K_c + K_z + K_int)])
        return {"apply_int": lambda R: K_int @ R, "z": z, "main": main, "full": full}
    col_c = _rbf_column(candidate_2d, (candidate_2d ** 2).sum(1), bw_c)
    col_z = _rbf_column(Z_scaled, (Z_scaled ** 2).sum(1), bw_z)
    ones = np.ones(n)
    z, main, full, inter = run_tasks([
        lambda: kernel_spectrum(n, ones, col_z, None),
        lambda: kernel_spectrum(n, 2 * ones, lambda p: col_c(p) + col_z(p), None),
        lambda: kernel_spectrum(n, 3 * ones, lambda p: col_c(p) + col_z(p) + col_c(p) * col_z(p), None),
        lambda: kernel_spectrum(n, ones, lambda p: col_c(p) * col_z(p), None),
    ])
    e_i, V_i, _ = inter
    return {"apply_int": lambda R: V_i @ (e_i[:, None] * (V_i.T @ R)), "z": z, "main": main, "full": full}


def _ridge_residuals(spectrum: tuple, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Residuals of kernel ridge regression of centred y on K = V diag(e) V':
    y - K (K + lambda I)^-1 y = y - V diag(e / (e + lambda)) V' y (V may be a
    thin basis)."""
    e, V = spectrum[0], spectrum[1]
    return y - V @ ((e / (e + ridge_lambda))[:, None] * (V.T @ y))


def _reml(spectrum: tuple, y: np.ndarray) -> float:
    """REML penalty of kernel ridge regression of centred y on the kernel."""
    e, V = spectrum[0], spectrum[1]
    P, outside = project(V, y)
    return reml_lambda_from_spectrum(e, P, n_total=y.shape[0], null_energy=outside)


def kernel_interaction_score_statistic(
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray,
        ridge_lambda: float | None = None,
        spectra: dict | None = None,
        lambda_z: float | None = None,
) -> tuple[float, float, float]:
    """Kernel-machine score statistic for a candidate-by-conditioning interaction,
    with partial and semipartial R^2 effect sizes.

    K_main = K_c + K_z is the additive null model, fitted by kernel ridge with a
    REML-estimated penalty unless ridge_lambda is given. The score is
    trace(r' K_int r) with K_int = K_c * K_z (Hadamard product), which reduces
    exactly to the scalar quadratic form when obs has one column. Effect sizes
    come from an augmented fit with K_main + K_int at the same penalty, as
    shares of generalized variance (kernels.variance_shares): the partial
    effect is the interaction's share of what the additive model leaves
    unexplained; the semipartial effect is its share of what the
    conditioning set leaves unexplained in obs, removed by kernel ridge
    regression on K_z (REML penalty), as the nonlinear tier removes it
    (docs/methods.md, p-values and effect sizes). spectra (interaction_spectra)
    may be passed to reuse the kernels. lambda_z is the penalty of the
    regression on the conditioning set alone (default: ridge_lambda when
    given, else REML).

    Returns (score, partial_effect_size, semipartial_effect_size).
    """
    spectra = spectra or interaction_spectra(candidate, conditioning)
    obs_2d = as2d(obs)
    obs_c = obs_2d - obs_2d.mean(axis=0)  # per-column centring

    fixed_lambda = ridge_lambda
    if ridge_lambda is None:
        ridge_lambda = _reml(spectra["main"], obs_c)

    resid_null = _ridge_residuals(spectra["main"], obs_c, ridge_lambda)
    if lambda_z is None:
        lambda_z = fixed_lambda if fixed_lambda is not None else _reml(spectra["z"], obs_c)
    resid_z = _ridge_residuals(spectra["z"], obs_c, lambda_z)

    # trace(r' K_int r) without forming the (q, q) matrix
    score = float(np.sum(resid_null * spectra["apply_int"](resid_null)))
    score = max(score, 0.0)

    resid_aug = _ridge_residuals(spectra["full"], obs_c, ridge_lambda)
    partial_es, semipartial_es = variance_shares(resid_z, resid_null, resid_aug)

    return score, partial_es, semipartial_es


class StateIndependenceTest(FunctionalFormTest):
    """Kernel score test of whether one fixed-shape relationship adequately summarizes the candidate-obs
    relationship given the full joint conditioning set (H0: no effect
    modification). Supports multivariate candidate and obs.

    With empty conditioning the question is vacuous and a zero statistic is
    returned (never rejected). ridge_lambda=None estimates the penalties by
    REML on the observed data, once per test, and holds them fixed in every
    draw (module docstring). The kernels and their eigendecompositions
    depend only on the candidate and the conditioning set, so they are
    computed once per test, and each draw costs O(n m) for spectra of m
    components. max_n caps the sample used (the shared kernel sample cap).
    """

    _supports_multivariate = True
    independent_multipliers = "rademacher"
    # Rademacher weights where samples are independent: with |w| = 1 each residual keeps its own square, so the
    # resampled score does not inherit the multipliers' extra kurtosis (Davidson and Flachaire, 2008).

    def __init__(self, alpha: float = 0.05, effect_size_floor: float = 0.02,
                 ridge_lambda: float | None = None, max_n: int = 2000, run_length: int = DEFAULT_RUN_LENGTH):
        super().__init__(alpha=alpha, effect_size_floor=effect_size_floor, run_length=run_length)
        self.ridge_lambda = ridge_lambda
        self.max_n = max_n

    @property
    def name(self) -> str:
        return "StateIndependenceTest(kernel-interaction-score)"

    def _classical_p_value(self, statistic: float, df_num: int, df_den: int) -> float:
        return float("nan")  # Davies' (1980) exact reference not implemented

    def _spectra(self, candidate, conditioning) -> dict:
        """Spectra for these exact arrays, reused across draws of one test."""
        cached = self.__dict__.get("_spectra_cache")
        if cached is None or cached[0] is not candidate or cached[1] is not conditioning:
            cached = (candidate, conditioning, interaction_spectra(candidate, conditioning))
            self.__dict__["_spectra_cache"] = cached
        return cached[2]

    def _penalties(self, candidate, obs, conditioning) -> tuple[float, float]:
        """(penalty of the additive null model, penalty of the regression on
        the conditioning set alone): the configured ridge_lambda for both, or
        else the REML estimates fixed for the current test (run), or else
        REML on these data."""
        if self.ridge_lambda is not None:
            return self.ridge_lambda, self.ridge_lambda
        fixed = self.__dict__.get("_fixed_penalties")
        if fixed is not None:
            return fixed
        spectra = self._spectra(candidate, conditioning)
        obs_2d = as2d(obs)
        obs_c = obs_2d - obs_2d.mean(axis=0)
        return _reml(spectra["main"], obs_c), _reml(spectra["z"], obs_c)

    def _compute_statistic(
            self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
    ) -> tuple[float, int, int, float, float]:
        if conditioning is None or np.asarray(conditioning).size == 0:
            return 0.0, 1, 1, 0.0, 0.0  # no conditioning: effect modification is vacuous
        lam, lam_z = self._penalties(candidate, obs, conditioning)
        score, partial_es, semipartial_es = kernel_interaction_score_statistic(
            candidate, obs, conditioning, ridge_lambda=lam,
            spectra=self._spectra(candidate, conditioning), lambda_z=lam_z,
        )
        return score, 1, 1, partial_es, semipartial_es  # degrees of freedom unused

    def _null_fit(self, candidate, obs, conditioning):
        """Null model: additive kernel ridge of obs on the candidate and the
        conditioning set (K_c + K_z), with no interaction."""
        obs_2d = as2d(obs)
        if conditioning is None or np.asarray(conditioning).size == 0:
            return np.broadcast_to(obs_2d.mean(axis=0), obs_2d.shape).reshape(obs.shape).copy()
        obs_c = obs_2d - obs_2d.mean(axis=0)
        lam, _ = self._penalties(candidate, obs, conditioning)
        resid_null = _ridge_residuals(self._spectra(candidate, conditioning)["main"], obs_c, lam)
        return (obs_2d - resid_null).reshape(obs.shape)

    def _null_leverage(self, candidate, obs, conditioning):
        """1 - Var(r_t) / sigma^2 for the residuals r = (I - H) y of the null
        model's fit (centring, then kernel ridge at the fixed penalty), so
        that the bootstrap divides each residual by its standard deviation
        under the fit, sqrt(diag((I - H)(I - H)')) (module docstring)."""
        n = as2d(candidate).shape[0]
        if conditioning is None or np.asarray(conditioning).size == 0:
            return np.full(n, 1.0 / n)  # residuals about the mean
        lam, _ = self._penalties(candidate, obs, conditioning)
        e, V = self._spectra(candidate, conditioning)["main"][:2]
        # Residual maker after centring, R = (I - H)(I - 11'/n) with H = V diag(d) V':
        # diag(R R') = diag(C) - 2 diag(H C) + diag(H C H), from V in O(n m).
        d = e / (e + lam)
        h1 = V @ (d * V.sum(axis=0))                 # H 1
        diag_h = (V ** 2) @ d
        diag_hh = (V ** 2) @ (d ** 2)
        diag_rr = (1.0 - 1.0 / n) - 2.0 * (diag_h - h1 / n) + (diag_hh - h1 ** 2 / n)
        return 1.0 - diag_rr

    def run(
            self,
            candidate: np.ndarray,
            obs: np.ndarray,
            conditioning: np.ndarray | None,
            time_positions: np.ndarray | None = None,
            min_temporal_lag: int = 0,
            n_permutations: int = 500,
            rng: np.random.Generator | None = None,
            grid_step: float | None = None,
    ) -> FunctionalFormTestResult:
        """Same contract as FunctionalFormTest.run, except that when n > max_n one
        evenly spaced subsample of max_n points is drawn first (the bootstrap
        accounts for its dependence) and used for the observed statistic and
        every draw.
        Subsampling inside each draw would compare the observed and null
        statistics on different data.
        """
        rng = rng or np.random.default_rng()
        candidate = as2d(np.asarray(candidate, dtype=np.float64))
        obs = as2d(np.asarray(obs, dtype=np.float64))
        n = candidate.shape[0]

        if time_positions is None:
            time_positions = np.arange(n)
        else:
            time_positions = np.asarray(time_positions)
            if time_positions.shape[0] != n:
                raise ValueError("time_positions must have the same length as candidate")

        has_conditioning = conditioning is not None and np.asarray(conditioning).size > 0
        was_subsampled = False
        natural_independent_n: int | None = None
        block_min_temporal_lag = min_temporal_lag
        if has_conditioning and grid_step is not None:  # long timestep records; event samples are small
            idx, positions, step, natural_independent_n = grid_subsample(
                time_positions, grid_step, min_temporal_lag, self.max_n, rng, at_lag_spacing=False)
            was_subsampled = idx.size < n
            if was_subsampled:
                candidate = candidate[idx]
                obs = obs[idx]
                conditioning = as2d(conditioning)[idx]
                time_positions, grid_step = positions, step

        try:
            if has_conditioning and self.ridge_lambda is None:
                # Penalties estimated once, on the observed data, and held fixed in every draw.
                self.__dict__["_fixed_penalties"] = self._penalties(candidate, obs, conditioning)
            result = super().run(
                candidate, obs, conditioning,
                time_positions=time_positions, min_temporal_lag=block_min_temporal_lag,
                n_permutations=n_permutations, rng=rng, grid_step=grid_step,
            )
        finally:
            self.__dict__.pop("_spectra_cache", None)
            self.__dict__.pop("_fixed_penalties", None)
        result.dependence_aware_subsampling_used = was_subsampled
        result.n_after_subsampling = candidate.shape[0]
        result.natural_independent_sample_size = natural_independent_n
        return result


def make_state_independence_adequacy_test(
        alpha: float = 0.05, effect_size_floor: float = 0.02,
        ridge_lambda: float | None = None, max_n: int = 2000, run_length: int = DEFAULT_RUN_LENGTH,
) -> AdequacyTestFunc:
    """Nonlinear-tier adequacy check (StateIndependenceTest)."""
    return as_adequacy_test(StateIndependenceTest(
        alpha=alpha, effect_size_floor=effect_size_floor, ridge_lambda=ridge_lambda, max_n=max_n,
        run_length=run_length))

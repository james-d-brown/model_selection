"""
Linear adjustment test: can the conditioning set be removed linearly? (linear tier).

The linear tier assumes the selected formulations influence the candidate and the
observations linearly. For each column y of the candidate and of the
observations, a kernel score test (Liu, Lin and Ghosh, 2007) asks whether
the residuals of the least-squares regression of y on Z depend on Z in any
smooth nonlinear way: the statistic is r'Kr / r'r, with r the residuals and
K a radial basis function kernel on the standardized Z (median-heuristic
bandwidth, Gretton et al., 2012). Because K spans smooth functions of all of
Z, the test detects nonlinearity in any direction of Z, including directions
in which the linear fit has no slope, which tests built on powers of the
fitted values (RESET, Ramsey, 1969) cannot see.

The null distribution is built by a wild bootstrap of the residuals under
the linear model (Wu, 1986; Mammen, 1993): each draw multiplies the
residuals, each divided by sqrt(1 - h_t) with h_t its leverage (the HC2
correction; Davidson and Flachaire, 2008), by multipliers of mean zero and
variance one, correlated over the dependence lag (Shao, 2010), refits on Z
and recomputes the statistic. It is valid when the residuals' spread varies
with Z. On samples on a regular time
grid the bandwidth of the multipliers is the Politis–White block length of
the statistic's per-sample terms r_t (K r)_t. The effect size is the share of
the residuals' variation that a kernel ridge regression on Z explains
(penalty by restricted maximum likelihood), bias-corrected by its mean over
the null draws. On long records the test runs on max_n evenly spaced points
(grid_subsample; the shared kernel sample cap, SelectionConfig.nonlinear_max_n);
the bootstrap accounts for the dependence that remains between them. Beyond
the rank cap the radial basis function kernel is a pivoted-Cholesky factor
(regression.kernel_residuals), used through its thin spectrum: the statistic,
the penalty, the effect size and every bootstrap draw use the same factor, so
an approximation at the cap changes which departures the test is most
sensitive to, not its calibration. Below the cap every quantity is the one
computed from the kernel's full eigendecomposition.

A column rejects when its p-value is below alpha divided by the number of
columns tested (Bonferroni) and its bias-corrected effect reaches the
effect-size floor.

See docs/methods.md, Linear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.adequacy.base import AdequacyTestOutcome
from model_selection.statistics.adequacy.kernels import (
    kernel_spectrum, project, rbf_kernel, reml_lambda_from_spectrum)
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights
from model_selection.statistics.regression.kernel_residuals import _median_bandwidth, _rbf_column, standardize
from model_selection.statistics.significance.base import bias_correct_association, permutation_p_value
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block
from model_selection.statistics.significance.resampling import grid_subsample
from model_selection.utils import as2d

DRAW_CHUNK = 128  # bootstrap draws processed at a time


def _kernel_scores(R: np.ndarray, eigvals: np.ndarray, V: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """For residual vectors in the columns of R (n, m): the score r'Kr / r'r
    and the share of r's variation explained by the kernel ridge fit on K
    with penalty lam, from K's spectrum (V may be a thin basis: the part of r
    outside it is left unexplained)."""
    P, outside = project(V, R)
    total = np.sum(P ** 2, axis=0) + outside
    total = np.where(total > 0, total, 1.0)
    score = (eigvals[:, None] * P ** 2).sum(axis=0) / total
    left = (((lam / (eigvals + lam))[:, None] * P) ** 2).sum(axis=0) + outside
    share = 1.0 - left / total
    return score, np.clip(share, 0.0, 1.0)


def linear_specification_test(
        columns: np.ndarray,
        conditioning: np.ndarray,
        alpha: float,
        effect_size_floor: float,
        n_permutations: int,
        rng: np.random.Generator,
        time_positions: np.ndarray,
        min_temporal_lag: int,
        column_labels: list[str] | None = None,
        run_length: int = DEFAULT_RUN_LENGTH,
        grid_step: float | None = None,
        n_candidate_columns: int | None = None,
        max_n: int = 2000,
        interaction_with: np.ndarray | None = None,
) -> AdequacyTestOutcome:
    """Test every column of `columns` for nonlinear dependence on the
    conditioning set.

    The first n_candidate_columns columns belong to the candidate and the
    rest to the observations. Linear partialling leaves the linear tier's test valid
    when either the candidate's or the observations' conditional mean on Z is
    linear, because its error depends on the product of the two regressions'
    errors (Shah and Bühlmann, 2023); the tier is therefore inadequate only when
    a candidate column and an observation column both reject. Without
    n_candidate_columns, any rejecting column makes it inadequate.

    column_labels : one readable name per column, used in the decision
        reason (default "column 1", "column 2", ...).
    max_n : largest sample tested; longer records on a regular grid are
        subsampled to max_n evenly spaced points (grid_subsample; the
        bootstrap accounts for the dependence that remains).
    interaction_with : covariates already removed from every series by
        kernel regression (the partially linear model of the high-flow
        linear tier). The test then has two parts, each at alpha / 2: a
        curvature part, with the radial basis function kernel on the
        conditioning set, and an interaction part, whose kernel is the
        product of linear kernels on the conditioning set and on these
        covariates, so it tests every product of a conditioning column and a
        covariate together (Maity and Lin, 2011). A column departs when
        either part rejects for it.
    """
    Y = as2d(np.asarray(columns, dtype=float))
    Z = as2d(np.asarray(conditioning, dtype=float))
    W_int = None if interaction_with is None else as2d(np.asarray(interaction_with, dtype=float))
    if column_labels is not None and len(column_labels) != Y.shape[1]:
        raise ValueError(f"column_labels has {len(column_labels)} entries for {Y.shape[1]} columns")
    time_positions = np.asarray(time_positions)
    if grid_step is not None and Y.shape[0] > max_n:
        idx, time_positions, grid_step, _ = grid_subsample(time_positions, grid_step, min_temporal_lag, max_n, rng,
                                                           at_lag_spacing=False)
        Y, Z = Y[idx], Z[idx]
        W_int = None if W_int is None else W_int[idx]
    n = Y.shape[0]
    design = np.column_stack([np.ones(n), Z])
    Q, _ = np.linalg.qr(design)
    hc2 = 1.0 / np.sqrt(np.clip(1.0 - np.sum(Q ** 2, axis=1), 1e-12, None))
    Zs = standardize(Z)
    bw = _median_bandwidth(Zs)
    sq = (Zs ** 2).sum(1)
    rbf = _rbf_column(Zs, sq, bw)
    # Each kernel's spectrum: the full eigendecomposition within the rank cap,
    # else a pivoted-Cholesky factor's thin spectrum. The interaction kernel,
    # a product of linear kernels, has an exact factor of k_z * k_w columns.
    spectra = [("curvature", lambda: kernel_spectrum(n, np.ones(n), rbf, lambda: rbf_kernel(Zs, bw)))]
    if W_int is not None:
        Ws = standardize(W_int)

        def interaction_spectrum():
            F = (Zs[:, :, None] * Ws[:, None, :]).reshape(n, -1)
            if F.shape[1] >= n:
                e, V = np.linalg.eigh(F @ F.T)
                return np.clip(e, 0.0, None), V, 0.0
            U, sv, _ = np.linalg.svd(F, full_matrices=False)
            return sv ** 2, U, 0.0
        spectra.append(("interaction", interaction_spectrum))
    from model_selection.parallel import run_tasks
    kernels = [(part, sp) for (part, _), sp in zip(spectra, run_tasks([f for _, f in spectra]))]
    blocks = []

    def column_outcomes(spectrum, level):
        eigvals, V, _ = spectrum
        found = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            resid = y - Q @ (Q.T @ y)
            P, outside = project(V, resid[:, None])
            lam = reml_lambda_from_spectrum(eigvals, P, n_total=n, null_energy=outside)
            observed, semipartial = (float(v[0]) for v in _kernel_scores(resid[:, None], eigvals, V, lam))
            K_resid = V @ (eigvals * P[:, 0])
            resolved = resolve_block(time_positions, min_temporal_lag, [(resid, K_resid)], run_length, grid_step)
            blocks.append(resolved)
            null_stats = np.empty(n_permutations)
            null_semi = np.empty(n_permutations)
            for start in range(0, n_permutations, DRAW_CHUNK):
                stop = min(start + DRAW_CHUNK, n_permutations)
                W = dependent_wild_weights(time_positions, resolved.length, stop - start, rng)
                R = (W * (resid * hc2)[None, :]).T
                R = R - Q @ (Q.T @ R)
                null_stats[start:stop], null_semi[start:stop] = _kernel_scores(R, eigvals, V, lam)
            p = permutation_p_value(observed, null_stats)
            corrected, _ = bias_correct_association(semipartial, null_semi)
            found.append((p < level and corrected >= effect_size_floor, p, j, semipartial, corrected))
        return found

    level = alpha / len(kernels) / Y.shape[1]
    by_part = [(part, column_outcomes(spectrum, level)) for part, spectrum in kernels]
    # A column's outcome: its departing part if any (the smaller p-value first), else its smallest p-value.
    outcomes, parts = [], []
    for j in range(Y.shape[1]):
        options = sorted(((o[j], part) for part, o in by_part), key=lambda op: (not op[0][0], op[0][1]))
        outcomes.append(options[0][0])
        parts.append(options[0][1])

    def name(j):
        label = column_labels[j] if column_labels is not None else f"column {j + 1}"
        return f"{label} ({parts[j]})" if len(kernels) > 1 else label

    def strongest(group):
        return min(group, key=lambda o: (not o[0], o[1])) if group else None

    if n_candidate_columns is None:
        sides = [outcomes]
    else:
        sides = [outcomes[:n_candidate_columns], outcomes[n_candidate_columns:]]
    worst_by_side = [strongest(side) for side in sides]
    rejected = all(w is not None and w[0] for w in worst_by_side)
    worst = strongest(outcomes)
    _, p, j, semipartial, corrected = worst
    where = name(j)
    if rejected:
        detail = "; ".join(f"{name(w[2])}: p={w[1]:.4g} < {level:.4g}, bias-corrected semipartial "
                           f"R^2={w[4]:.4g} >= {effect_size_floor:.4g}" for w in worst_by_side)
        both = " (both the candidate and the observations)" if n_candidate_columns is not None else ""
        reason = f"rejected: the selected formulations cannot be removed linearly{both}: {detail}"
    elif n_candidate_columns is not None and worst[0]:
        reason = (f"not rejected: {where} departs from linearity (p={p:.4g}, bias-corrected semipartial "
                  f"R^2={corrected:.4g}), but the other side's regression is linear, which keeps the "
                  "linear tier's test valid")
    else:
        reason = (f"not rejected: strongest departure in {where} (p={p:.4g}, "
                  f"bias-corrected semipartial R^2={corrected:.4g})")
    return AdequacyTestOutcome(
        tier_adequate=not rejected, decision_reason=reason, p_value=p,
        semipartial_effect_size=semipartial, bias_corrected_semipartial_effect_size=corrected,
        block_length=max(b.length for b in blocks),
        dependence={
            "block_length": max(b.length for b in blocks), "estimated": blocks[0].estimated,
            "decorrelation_lag": (max(b.decorrelation_lag for b in blocks) if blocks[0].estimated else None),
            "dependence_decays": all(b.decays for b in blocks),
            "n_blocks": (min(b.n_blocks for b in blocks) if all(b.n_blocks is not None for b in blocks) else None),
            "few_blocks": any(b.few_blocks for b in blocks),
            "n_samples": blocks[0].n_samples,
            "too_few_to_assess": any(b.too_few_to_assess for b in blocks),
        },
    )

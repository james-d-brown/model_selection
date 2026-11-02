"""
Greedy forward selection of model formulations by conditional independence
testing (see docs/methods.md, The cascade).

Each round scores every remaining candidate with the tier's test given the
formulations selected so far, under one shared resampling. The candidate
with the largest effect size is the round's winner. In order:

  1. Structural check (linear tier): if the tier is undefined for the
     round's dimensions, the trial stops: as ill-posed in round 1, otherwise
     with the formulations already selected.
  2. Linear adjustment test (linear tier, whenever the conditioning set is
     non-empty, so from round 1 when covariates are conditioned on): can the
     selected formulations and covariates be removed linearly? If not, the
     trial is discarded (tier_rejected).
  3. Acceptance test: the winner's effect size is compared with the null
     distribution of the maximum effect size across all remaining
     candidates (Westfall and Young, 1993). Acceptance requires both conditions:
     p < alpha, and the bias-corrected effect size (observed minus the mean
     of the winner's own null draws) reaching the effect-size floor.
  4. Candidate linearity and effect-modification tests (linear tier) or
     candidate effect-modification test (nonlinear tier): is the candidate
     related to the observations in the form the tier assumes? Recorded for
     information when the winner was accepted. When it was not, every
     remaining candidate is tested, the winner first, at alpha / (m k) for m
     remaining candidates and k tests (Bonferroni), and the first rejection
     discards the trial so the cascade can escalate.

Selection stops at the first winner not accepted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from model_selection.selection.tiers import SelectionSample, Tier
from model_selection.statistics.power import effective_sample_size, format_share, minimum_detectable_r2
from model_selection.statistics.conditional_independence import LinearResidualTest
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH
from model_selection.statistics.significance.base import bias_correct_association, permutation_p_value
from model_selection.utils import as2d

logger = logging.getLogger(__name__)


@dataclass
class SelectionStep:
    """One tested winner.

    association is the winner's statistic on the tier's scale given the
    selection so far; effect is the effect size the acceptance test compared
    with the floor (the association minus its null mean in the linear and
    free-form tiers, the cross-fitted magnitude in the nonlinear tier); p_value is calibrated against the maximum across the
    n_candidates_tested candidates. diagnostics reports the test's assumption
    diagnostics (docs/methods.md, Diagnostics). detectable_r2
    is the smallest set-correlation R^2 the linear tier could detect with 80%
    power at this step (Cohen, 1988) with effective_sample_size independent
    samples, matched to the linear tier's resampled null on this step's sample
    (docs/methods.md, Detectable effect and the meaningful-effect
    threshold). It is an optimistic bound in every tier: it ignores the
    calibration against the maximum over candidates and the effect-size gate,
    and the other tiers have less power. block_length is the block length (or
    multiplier bandwidth) the significance test's resampling used, and
    dependence how it was obtained (docs/methods.md, Dependence between samples).
    """
    formulation: str
    association: float
    p_value: float
    accepted: bool
    n_selected_before: int
    n_candidates_tested: int
    dependence_lag: int
    null_mean: float = 0.0
    null_std: float = 0.0
    effect: float = 0.0
    magnitude_unsubstantiated: bool = False
    diagnostics: dict = field(default_factory=dict)
    adequacy: list[dict] = field(default_factory=list)
    other_candidates_checked: list[dict] = field(default_factory=list)
    tier: str | None = None
    """The tier whose test made this step (earlier tiers' accepted steps carry over after a
    power-only escalation)."""
    cmi_calibration: dict | None = None
    samples_per_dimension: float = float("inf")
    low_samples_per_dimension: bool = False
    detectable_r2: float | None = None
    effective_sample_size: float | None = None
    block_length: int | None = None
    dependence: dict | None = None


@dataclass
class SelectionResult:
    """Outcome of one greedy selection trial.

    unconditional_association holds every candidate's effect size given no
    selection (round 1). When tier_rejected or ill_posed is True the trial is
    invalid and `selected` is []. ill_posed is True only when round 1 fails the
    tier's sample-size requirement; a later round that fails it ends the trial
    with the formulations already selected, and ill_posed_reason records why.
    diagnostic_plot_paths lists the adequacy plots written.
    """
    selected: list[str] = field(default_factory=list)
    trajectory: list[SelectionStep] = field(default_factory=list)
    stopped_reason: str = ""
    unconditional_association: dict[str, float] = field(default_factory=dict)
    unconditional_effect: dict[str, float] = field(default_factory=dict)
    """Every candidate's effect size in round 1, given only the sample's
    covariates: the share of the observations' variation each formulation
    explains on its own, on the analysis scale."""
    tier_rejected: bool = False
    tier_rejection_reason: str | None = None
    diagnostic_plot_paths: list[str] = field(default_factory=list)
    ill_posed: bool = False
    ill_posed_reason: str | None = None
    equivalence: list[dict] = field(default_factory=list)
    equivalent_sets: list[dict] = field(default_factory=list)
    equivalent_sets_summary: str = ""
    next_tier_unsupported: str | None = None
    """Set on a rejected trial whose escalation leads to a tier that cannot
    be used on its sample (not calibrated); the cascade then does not run it."""
    escalation_continues: bool = False
    """True when the trial was rejected by a power-only (relationship) test:
    its acceptances remain valid, `selected` keeps them, and the next tier
    continues from them rather than restarting."""
    escalation_test: dict | None = None
    """On a rejected trial, the test that rejected the tier: round,
    candidate, check, p-value, level, resampling diagnostics, and the round's
    smallest detectable R^2 in this tier (for a power-only rejection)."""
    escalated_from: dict | None = None
    """Set by the cascade when the trial's last round was first tested in the
    previous tier, before a power-only escalation: that tier's name, the round
    and its smallest detectable R^2 for the round, to report alongside this
    tier's."""


def escalated_from_text(selection: "SelectionResult") -> str:
    """For a trial whose last round was first tested in the previous tier
    before a power-only escalation, that tier's smallest detectable R^2 for
    the round, which is usually more informative than the escalated tier's;
    "" otherwise."""
    prior = selection.escalated_from
    last = selection.trajectory[-1] if selection.trajectory else None
    if not prior or last is None or last.accepted or prior.get("detectable_r2") is None:
        return ""
    return (f"In the {prior['tier']} tier, which tested this round before the escalation, contributions below "
            f"R^2 = {format_share(prior['detectable_r2'])} cannot be excluded with 80% power.")


def _stack_selected(candidate_series: dict, selected: list[str],
                    covariates: np.ndarray | None = None) -> np.ndarray | None:
    """The conditioning set: the sample's covariates and the selected
    formulations' values; None when both are empty."""
    parts = ([] if covariates is None else [as2d(covariates)]) + [as2d(candidate_series[s]) for s in selected]
    return np.hstack(parts) if parts else None


def _calibration_summary(calibration: dict) -> str:
    return (f"it accepted {calibration['false_acceptances']} of {calibration['n_surrogates']} surrogate candidates "
            "that carry no information beyond the conditioning set (binomial p="
            f"{calibration['p_value']:.3g} against the level alpha)")


def gate(observed: float, null_column: np.ndarray, null_max: np.ndarray, alpha: float,
         effect_size_floor: float, effect: float | None = None) -> tuple[bool, float, float]:
    """Apply both acceptance gates: the p-value of `observed` against the
    maximum null, and the effect size against the floor. The effect size is
    `effect` when given (the test's magnitude on the floor's scale),
    otherwise the observed value minus the mean of its own null draws.
    Returns (accepted, p_value, effect)."""
    p_value = permutation_p_value(observed, null_max)
    if effect is None:
        effect, _ = bias_correct_association(observed, null_column)
    accepted = bool(p_value < alpha and effect >= effect_size_floor)
    return accepted, float(p_value), float(effect)


def _column_labels(series: str, values: np.ndarray, feature_names: tuple[str, ...] | None) -> list[str]:
    """Readable names for the columns of one series, e.g.
    "candidate 'x', feature 2 (PC2)"; a single-column series is named alone."""
    d = as2d(values).shape[1]
    if d == 1 and not feature_names:
        return [series]
    names = list(feature_names) if feature_names and len(feature_names) == d else [None] * d
    return [f"{series}, feature {j + 1}" + (f" ({nm})" if nm else "") for j, nm in enumerate(names)]


def _sentence(reason: str) -> str:
    """A test's decision reason as a sentence."""
    reason = reason.strip()
    return (reason[:1].upper() + reason[1:] + ("" if reason.endswith(".") else ".")) if reason else ""


def greedy_forward_selection(
        sample: SelectionSample,
        tier: Tier,
        n_permutations: int = 500,
        alpha: float = 0.05,
        max_selected: int | None = None,
        rng: np.random.Generator | None = None,
        min_samples_per_dimension: float = 10.0,
        carried_steps: list[SelectionStep] | None = None,
) -> SelectionResult:
    """Run one greedy selection trial under `tier` on `sample`. Steps with
    fewer than min_samples_per_dimension samples per dimension (candidate,
    observations and selected set together) are flagged. carried_steps are
    accepted steps of an earlier tier that this trial continues from (after a
    power-only escalation): their formulations start the selection, and the
    steps start the trajectory."""
    rng = rng or np.random.default_rng()
    candidates, obs = sample.candidates, sample.obs
    n = len(obs)
    if sample.time_positions is not None and len(sample.time_positions) != n:
        raise ValueError(
            f"time_positions has length {len(sample.time_positions)} but obs has length {n}; "
            "they must index the same samples."
        )
    time_positions = sample.time_positions if sample.time_positions is not None else np.arange(n)
    lag = sample.dependence_lag

    carried_steps = list(carried_steps or [])
    selected: list[str] = [st.formulation for st in carried_steps]
    remaining = [c for c in candidates if c not in selected]
    trajectory: list[SelectionStep] = list(carried_steps)
    plots: list[str] = []
    unconditional: dict[str, float] = {}
    unconditional_effect: dict[str, float] = {}
    stopped_reason = "no candidates remaining"

    def finish(**kw) -> SelectionResult:
        return SelectionResult(selected=kw.pop("selected", selected), trajectory=trajectory,
                               stopped_reason=stopped_reason, unconditional_association=unconditional,
                               unconditional_effect=unconditional_effect,
                               diagnostic_plot_paths=plots, **kw)

    while remaining:
        if max_selected is not None and len(selected) >= max_selected:
            stopped_reason = f"reached max_selected={max_selected}"
            break
        z = _stack_selected(candidates, selected, sample.covariates)
        round_no = len(selected) + 1

        if tier.structural_check is not None:
            k_dim = 0 if z is None else z.shape[1]
            d_c = max(as2d(candidates[c]).shape[1] for c in remaining)
            outcome = tier.structural_check(n, d_c, as2d(obs).shape[1], k_dim)
            if not outcome.well_posed:
                if not selected:
                    stopped_reason = f"ill-posed at round {round_no} (structural check): {outcome.reason}"
                    logger.info(f"[!] {stopped_reason}; trial discarded.")
                    return finish(selected=[], ill_posed=True, ill_posed_reason=outcome.reason)
                # The earlier rounds' tests met the requirement; only the next
                # test cannot be made, so selection ends with what it has.
                stopped_reason = (f"round {round_no} could not be tested (sample-size requirement): "
                                  f"{outcome.reason}; selection ends with the formulations already selected")
                logger.info(f"[!] {stopped_reason}.")
                return finish(ill_posed_reason=outcome.reason)

        logger.info(f"[*] Round {round_no} ({tier.name} tier): testing {len(remaining)} candidates "
                    f"with {n_permutations} resamples.")
        result = tier.test.run({c: candidates[c] for c in remaining}, obs, z, n_permutations, rng,
                               time_positions, lag, grid_step=sample.grid_step)
        if round_no == 1:
            unconditional = {nm: float(v) for nm, v in zip(result.names, result.observed)}
        # The round's winner is the candidate with the largest effect size among
        # those whose statistic is significant against the null of the maximum
        # (all candidates when none is). The maximum null controls false
        # selections for whichever candidate is chosen (Westfall and Young,
        # 1993), and a winner that is not accepted then means no remaining
        # candidate would be. In the linear and free-form tiers the effect size
        # is the statistic less its chance level, so the order is the
        # statistic's; in the nonlinear tier it is a different quantity
        # (docs/methods.md, Acceptance test).
        null_max = result.null.max(axis=1)
        effects = {}
        for k, nm in enumerate(result.names):
            effects[nm] = tier.test.magnitude(candidates[nm], obs, z, time_positions, lag,
                                              float(result.observed[k]), result.null[:, k],
                                              grid_step=sample.grid_step)
        if round_no == 1:
            unconditional_effect = {nm: float(e[0]) for nm, e in effects.items()}
        significant = [nm for k, nm in enumerate(result.names)
                       if permutation_p_value(float(result.observed[k]), null_max) < alpha]
        winner = max(significant or result.names, key=lambda nm: (effects[nm][0], nm))
        w = result.index(winner)
        winner_values = candidates[winner]
        logger.info(f"[+] Round {round_no} winner: {winner} (effect size {effects[winner][0]:.4g}, "
                    f"statistic {result.observed[w]:.4g}).")

        round_label = (f"round{round_no}_{winner}_given_{'+'.join(selected)}" if selected
                       else f"round{round_no}_{winner}")
        def context(name: str, result: str | None = None):
            from model_selection.visualization.adequacy import PlotContext
            return PlotContext(tier=tier.name, round_no=round_no, candidate=name, selected=tuple(selected),
                               covariate_label=sample.covariate_label,
                               n_covariate_columns=0 if sample.covariates is None else as2d(sample.covariates).shape[1],
                               feature_names=sample.feature_names, result=result)

        def draw(check, name: str, result: str) -> None:
            if check.plot is not None:
                path = check.plot(candidates[name], obs, z, round_label, context(name, result))
                if path is not None:
                    plots.append(path)

        # Descriptive plots first, so a round whose adequacy check rejects still has them.
        for plot in tier.diagnostic_plots:
            path = plot(winner_values, obs, z, round_label, context(winner))
            if path is not None:
                plots.append(path)

        def run_checks(stage: str, name: str = winner, plot: bool = False) -> list[tuple]:
            values = candidates[name]
            done = []
            for check in tier.adequacy_checks:
                if check.stage != stage or (check.needs_conditioning and z is None):
                    continue
                extra = {}
                if check.labels_columns:
                    extra["column_labels"] = (_column_labels(f"candidate '{name}'", values, sample.feature_names)
                                              + _column_labels("observations", obs, sample.feature_names))
                outcome = check.test(values, obs, z, n_permutations=tier.adequacy_n_permutations,
                                     rng=rng, time_positions=time_positions, min_temporal_lag=lag,
                                     grid_step=sample.grid_step, **extra)
                done.append((check, outcome))
            return done

        def record(done: list[tuple], role: str) -> list[dict]:
            return [{"check": c.name, "adequate": o.tier_adequate, "role": role, "p_value": o.p_value,
                     "bias_corrected_semipartial_r2": o.bias_corrected_semipartial_effect_size,
                     "block_length": o.block_length, "dependence": o.dependence,
                     "decision_reason": o.decision_reason} for c, o in done]

        # Removal checks come first: a candidate cannot be judged if the selected
        # formulations have not been removed adequately.
        removal = run_checks("removal")
        for check, outcome in removal:
            draw(check, winner, _sentence(outcome.decision_reason) + (
                f" The {tier.name} tier's results may then be invalid, so selection moves to the next tier, which "
                "starts a new trial from an empty selection." if not outcome.tier_adequate else
                f" The {tier.name} tier can be used for this round."))
        for check, outcome in removal:
            if not outcome.tier_adequate:
                stopped_reason = (f"tier rejected at round {round_no}: the check for "
                                  f"{check.name} rejects for '{winner}' ({outcome.decision_reason})")
                logger.info(f"[!] {stopped_reason}; trial discarded, escalation required.")
                return finish(selected=[], tier_rejected=True, tier_rejection_reason=outcome.decision_reason,
                              escalation_test={"round": round_no, "candidate": winner, "check": check.name,
                                               "p_value": outcome.p_value, "dependence": outcome.dependence,
                                               "detectable_r2": None})

        null_column = result.null[:, w]
        effect, effect_null_mean = effects[winner]
        accepted, p_value, bias_corrected = gate(
            float(result.observed[w]), null_column, null_max, alpha, tier.effect_size_floor, effect=effect)
        magnitude_unsubstantiated = bool(bias_corrected < tier.effect_size_floor)
        # An acceptance by the CMI tier counts only where that tier is calibrated
        # on the sample (statistics.conditional_independence.calibration).
        calibration = None
        if accepted and tier.calibration is not None and z is not None:
            calibration = tier.calibration(winner_values, obs, z, time_positions, lag, rng.spawn(1)[0],
                                           grid_step=sample.grid_step)
            accepted = calibration["calibrated"]
        n_dims = as2d(winner_values).shape[1] + as2d(obs).shape[1] + (0 if z is None else z.shape[1])
        samples_per_dimension = n / n_dims
        k_y, k_x, k_c = as2d(obs).shape[1], as2d(winner_values).shape[1], 0 if z is None else z.shape[1]
        if isinstance(tier.test, LinearResidualTest):
            linear_null = null_column
        else:
            # The linear tier's null on this step's sample, from an independent
            # random stream so the selection's own draws are unchanged.
            linear_null = LinearResidualTest(run_length=getattr(tier.test, "run_length", DEFAULT_RUN_LENGTH)).run(
                {winner: winner_values}, obs, z, n_permutations, rng.spawn(1)[0], time_positions, lag,
                grid_step=sample.grid_step).null[:, 0]
        n_eff = effective_sample_size(n, linear_null, k_y, k_x, k_c)
        detectable_r2 = minimum_detectable_r2(n_eff, k_y, k_x, k_c, alpha)
        trajectory.append(SelectionStep(
            formulation=winner, association=float(result.observed[w]), p_value=p_value, accepted=accepted, tier=tier.name,
            n_selected_before=len(selected), n_candidates_tested=len(remaining), dependence_lag=lag,
            block_length=result.block_length, dependence=result.dependence,
            null_mean=float(np.mean(null_column)), null_std=float(np.std(null_column, ddof=1)),
            effect=bias_corrected,
            magnitude_unsubstantiated=magnitude_unsubstantiated,
            diagnostics=result.diagnostics.get(winner, {}),
            samples_per_dimension=samples_per_dimension,
            low_samples_per_dimension=bool(samples_per_dimension < min_samples_per_dimension),
            detectable_r2=detectable_r2,
            effective_sample_size=n_eff,
            cmi_calibration=calibration,
        ))
        logger.info(f"[+] {winner}: p={p_value:.4g}, effect {bias_corrected:.4g}, "
                    f"accepted={accepted}.")
        if calibration is not None and not calibration["calibrated"]:
            stopped_reason = (f"'{winner}' not selected at round {round_no}: the {tier.name} tier would accept it "
                              f"(p={p_value:.4g}, effect {bias_corrected:.4g}), but the tier is not calibrated on "
                              f"this sample: {_calibration_summary(calibration)}")
            logger.info(f"[!] {stopped_reason}.")
            trajectory[-1].adequacy = record(removal, "decision")
            break

        # Relationship checks come after the decision. If the winner was accepted,
        # they are recorded for information only: the tier has already detected
        # the contribution, and escalating could only lose power. If it was not
        # accepted, every remaining candidate is checked, since one whose
        # contribution is curved or changes sign with the state ranks low on
        # this tier's statistic: the tier's k relationship tests of the m
        # remaining candidates form one family, each at alpha / (m k)
        # (Bonferroni), and a departure in any of them escalates, so a
        # spurious escalation has probability at most alpha.
        step = trajectory[-1]
        if accepted:
            informative = run_checks("relationship")
            for check, outcome in informative:
                draw(check, winner, _sentence(outcome.decision_reason) + (
                    f" '{winner}' was accepted in this round, so this test is recorded for information only and "
                    "does not lead to the next tier."))
            step.adequacy = record(removal, "decision") + record(informative, "information")
        else:
            step.adequacy = record(removal, "decision")
            n_relationship = sum(c.stage == "relationship" and not (c.needs_conditioning and z is None)
                                 for c in tier.adequacy_checks)
            level = tier.adequacy_alpha / (len(remaining) * max(n_relationship, 1))
            statistic = dict(zip(result.names, result.observed))
            family = [winner] + sorted((nm for nm in remaining if nm != winner),
                                       key=lambda nm: (-statistic.get(nm, 0.0), nm))
            for nm in (family if n_relationship else []):
                departing = None
                for check, outcome in run_checks("relationship", nm):
                    departs = bool(not outcome.tier_adequate and outcome.p_value < level)
                    if nm == winner:
                        draw(check, winner, (
                            f"p = {outcome.p_value:.3g}, tested at {level:.3g}: the level α = "
                            f"{tier.adequacy_alpha:g} divided among the {len(remaining)} remaining "
                            f"candidate{'s' if len(remaining) > 1 else ''} and the tier's {n_relationship} "
                            f"test{'s' if n_relationship > 1 else ''} (Bonferroni). "
                            + (f"'{winner}' departs, so selection continues in the next tier from the current "
                               "selection." if departs else
                               f"No departure for '{winner}'; another remaining candidate may still depart.")))
                        entry = record([(check, outcome)], "decision")[0]
                        entry.update(adequate=not departs, level=level)
                        step.adequacy.append(entry)
                    else:
                        step.other_candidates_checked.append({
                            "candidate": nm, "check": check.name, "p_value": outcome.p_value, "level": level,
                            "bias_corrected_semipartial_r2": outcome.bias_corrected_semipartial_effect_size,
                            "dependence": outcome.dependence, "departs": departs})
                    if departs and departing is None:
                        departing = (check, outcome)
                if departing is None:
                    continue
                check, outcome = departing
                whose = f"'{winner}'" if nm == winner else f"the remaining candidate '{nm}'"
                stopped_reason = (f"tier rejected at round {round_no}: '{winner}' not accepted under the "
                                  f"{tier.name} tier, and the check for {check.name} rejects for {whose} at "
                                  f"{level:.3g} (Bonferroni across {n_relationship} test(s) of each of the "
                                  f"{len(remaining)} remaining candidates; "
                                  f"{outcome.decision_reason})")
                logger.info(f"[!] {stopped_reason}; escalation required, continuing from the selection so far.")
                unsupported = None
                # The CMI tier can be used only where it is calibrated on its
                # sample, for the candidate that leads there; otherwise no tier
                # whose assumptions hold can make the selection.
                if tier.escalation_guard is not None and z is not None:
                    step.cmi_calibration = tier.escalation_guard(
                        candidates[nm], obs, z, time_positions, lag, rng.spawn(1)[0], grid_step=sample.grid_step,
                        names=(nm, list(selected)))
                    if not step.cmi_calibration["calibrated"]:
                        unsupported = (f"the CMI tier, to which the departure leads, is not calibrated on its "
                                       f"sample ({_calibration_summary(step.cmi_calibration)})")
                # A relationship test concerns power only: the acceptances so far remain valid, so the next tier
                # continues from them.
                return finish(selected=list(selected), tier_rejected=True, tier_rejection_reason=outcome.decision_reason,
                              next_tier_unsupported=unsupported, escalation_continues=True,
                              escalation_test={"round": round_no, "candidate": nm, "check": check.name,
                                               "p_value": outcome.p_value, "level": level,
                                               "dependence": outcome.dependence,
                                               "detectable_r2": step.detectable_r2})

        if not accepted:
            failed = []
            if magnitude_unsubstantiated:
                if effect_null_mean is None:
                    failed.append(f"magnitude unsubstantiated (effect {bias_corrected:.4g}, the cross-fitted share of "
                                  "the observations' remaining generalized variance, < effect_size_floor "
                                  f"{tier.effect_size_floor:.4g})")
                else:
                    failed.append(f"magnitude unsubstantiated (bias-corrected effect {bias_corrected:.4g} < "
                                  f"effect_size_floor {tier.effect_size_floor:.4g}; raw {result.observed[w]:.4g}, "
                                  f"null mean {effect_null_mean:.4g})")
            if p_value >= alpha:
                failed.append(f"not statistically significant (p={p_value:.4g} >= alpha={alpha:.4g})")
            stopped_reason = f"'{winner}' not accepted: " + "; ".join(failed)
            if p_value < alpha:
                # Significant, but the point estimate of its size is below the floor.
                stopped_reason += (f". The contribution is statistically significant (p={p_value:.4g}), but its "
                                   "estimated size is below the effect-size floor")
            elif detectable_r2 is not None and detectable_r2 < 1.0:
                stopped_reason += (f". Contributions below R^2 = {format_share(detectable_r2)} "
                                   f"({format_share(100 * detectable_r2)}% of the remaining unexplained variation) "
                                   "cannot be excluded with 80% power")
            break
        selected.append(winner)
        remaining.remove(winner)
    else:
        if selected:
            stopped_reason = "all candidates selected"

    logger.info("[+] Completed greedy forward selection")
    return finish()

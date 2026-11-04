"""Result types of both pathways and their JSON-ready dictionaries."""
from __future__ import annotations

from dataclasses import dataclass, field

from model_selection.io.basin import OBS_KEY
from model_selection.event_detection.windows import ExpandedEvent
from model_selection.selection.greedy import SelectionResult
from model_selection.selection.config import DEFAULT_ADEQUACY_FLOOR
from model_selection.statistics.power import describe_detectable, format_share
from model_selection.statistics.significance.block_length import (
    MIN_ACF_N, MIN_BLOCKS, expected_variance_shortfall,
)

POOLED_RESULT_JSON_FILENAME_TEMPLATE = "{gage}_pooled_result.json"
HIGH_FLOW_RESULT_JSON_FILENAME_TEMPLATE = "{gage}_high_flow_result.json"


@dataclass
class PooledResult:
    """Outcome of the pooled pathway (one sample per timestep)."""
    gage: str
    n_timesteps: int
    zero_flow_fraction: dict[str, float]
    zero_flow_flag: bool
    selection: SelectionResult
    min_temporal_lag: int
    n_complete_timesteps: int | None = None
    missing_values: dict | None = None
    measure: str = "cmi"
    dependence_aware_subsampling_used: bool = False
    n_after_subsampling: int | None = None
    natural_independent_sample_size: int | None = None
    notes: list[str] = field(default_factory=list)
    result_json_path: str | None = None
    committed_tier: str | None = None
    attempted_tiers: list[dict] = field(default_factory=list)
    adequacy_floor: float | None = None
    seasonal_conditioning: dict | None = None
    seed: int | None = None
    meaningful_r2: float | None = None
    cmi_calibration: bool = True
    pearson_correlation: dict = field(default_factory=dict)
    """Each formulation's Pearson correlation with the observed flow at every
    complete time step; reported only, for comparison."""
    pearson_correlation_transformed: dict = field(default_factory=dict)
    """The same correlation on the series as transformed for selection (log
    flow less its calendar-day mean, as normal scores)."""


@dataclass
class HighFlowResult:
    """Outcome of the high-flow pathway (one sample per event).

    selection is None when there were too few usable events. n_components
    is the fitted component count. requires_closer_inspection is set when
    the basis sensitivity check finds the FPCA basis may confound the result.
    """
    gage: str
    n_timesteps: int
    zero_flow_fraction: dict[str, float]
    zero_flow_flag: bool
    selection: SelectionResult | None
    high_flow_power: dict
    high_flow_events: list[ExpandedEvent]
    n_raw_consensus_events: int
    n_components: int
    fpca_fit_source: str
    fpca_cumulative_explained_variance: float
    min_temporal_lag: int
    event_dependence: dict | None = None
    """Which events the tests treated as dependent: the rule, and for
    clusters their number and sizes, the gaps between consecutive clusters
    and, for clustering of arrivals, the run length (docs/methods.md,
    Dependence between samples)."""
    missing_values: dict | None = None
    seasonal_conditioning: dict | None = None
    pot_ratio: float = 2.0 / 3.0
    recession_ratio: float = 0.25
    wasp_metadata: dict | None = None
    fpca_component_orthogonality_error: float = 0.0
    fpca_regularization_lambda: float = 0.0
    fpca_n_grid: int | None = None
    fpca_explained_variance_ratio: list[float] = field(default_factory=list)
    fpca_reconstruction_fidelity: dict = field(default_factory=dict)
    peak_phase_spread: dict = field(default_factory=dict)
    reconstruction_fidelity_by_event: list[dict] = field(default_factory=list)
    basis_sensitivity_check: dict | None = None
    requires_closer_inspection: bool = False
    ensemble_representation: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    result_json_path: str | None = None
    diagnostic_plot_path: str | None = None
    event_gallery_path: str | None = None
    overview_diagnostic_path: str | None = None
    measure: str = "cmi"
    committed_tier: str | None = None
    attempted_tiers: list[dict] = field(default_factory=list)
    detectable_r2: float | None = None
    meaningful_r2: float | None = None
    power_stop_reason: str | None = None
    adequacy_floor: float | None = None
    seed: int | None = None
    cmi_calibration: bool = True
    pearson_correlation: dict = field(default_factory=dict)
    """Each formulation's Pearson correlation with the observed flow at the
    time steps inside the high-flow events; reported only, for comparison."""


@dataclass
class BasinResult:
    """Both pathways for one basin (high_flow is None if not run)."""
    gage: str
    pooled: PooledResult
    high_flow: HighFlowResult | None = None

    @property
    def zero_flow_flag(self) -> bool:
        return self.pooled.zero_flow_flag

    @property
    def zero_flow_fraction(self) -> dict[str, float]:
        return self.pooled.zero_flow_fraction


def _dependence_flags(where: str, sources: list[tuple[str, dict | None]]) -> list[dict]:
    """Flags from the resampling diagnostics of each (test, dependence) in
    sources, prefixed by where (round and formulation)."""
    flags = []
    for what, dep in sources:
        if dep and dep.get("crossfit_blocks_short"):
            flags.append({"code": "crossfit_blocks_short", "message": (
                f"{where}, {what}: the record spanned fewer than four dependence lags, so its cross-fitting "
                "blocks were shorter than one lag, or a block's fit kept its neighbouring blocks for lack of "
                "samples; dependence can leak between fitting and prediction, which may make the test reject "
                "too often.")})
        if dep and dep.get("cluster_terms_correlated"):
            flags.append({"code": "event_terms_correlated", "message": (
                f"{where}, {what}: the terms of the statistic (the products of the candidate's and the "
                "observations' least-squares residuals) of consecutive clusters of events were correlated "
                f"({dep.get('cluster_terms_correlation', 0.0):.2f}, standard error "
                f"{dep.get('cluster_terms_standard_error', 0.0):.2f} over {dep.get('n_clusters')} clusters), so "
                "dependence persists across events treated as independent, and the null may be too narrow "
                "(docs/methods.md, Dependence between samples).")})
        if dep and dep.get("low_rank_material"):
            flags.append({"code": "low_rank_material", "message": (
                f"{where}, {what}: a kernel factor reached the rank cap, and the cross-fitted residual sum of "
                f"squares changed by {dep.get('low_rank_rank_change', 0.0):.2g} (relative) between half the "
                "rank and the cap, so the effect size may still move by about twice that, as much as the "
                "effect-size floor; raise SelectionConfig.nonlinear_rank (docs/methods.md, Nonlinear tier).")})
        if dep and not dep.get("estimated") and dep.get("few_blocks") and dep.get("independent_units"):
            flags.append({"code": "few_blocks", "message": (
                f"{where}, {what}: the events were resampled as {dep.get('n_blocks', 0):.0f} independent clusters "
                f"of dependent events, fewer than the minimum of {MIN_BLOCKS} (a convention). With few clusters "
                "the wild cluster bootstrap can reject too often (MacKinnon and Webb, 2017; docs/methods.md, "
                "Dependence between samples).")})
        elif dep and not dep.get("estimated") and dep.get("few_blocks"):
            flags.append({"code": "few_blocks", "message": (
                f"{where}, {what}: the samples are not on a regular time grid "
                f"(events), and were resampled in blocks of the dependence lag, {dep.get('n_blocks', 0):.1f} "
                f"across their span, fewer than the minimum of {MIN_BLOCKS} (a convention). The null's spread "
                "is then estimated from few blocks and the test can reject too often; the record is short for "
                "its dependence lag. The block length is the lag, not an estimate, so the size of the error "
                "is not determined by the block count.")})
        if not dep or not dep.get("estimated"):
            continue
        if dep.get("independent") is False:
            flags.append({"code": "dependent_samples_permuted", "message": (
                f"{where}, {what}: the local permutation treats samples as "
                "exchangeable, but they remain autocorrelated (decorrelation lag "
                f"{dep.get('decorrelation_lag')} timesteps); the null may be too narrow.")})
            continue
        if dep.get("few_blocks"):
            flags.append({"code": "few_blocks", "message": (
                f"{where}, {what}: the block resampling rested on "
                f"{dep.get('n_blocks', 0):.0f} blocks, fewer than the minimum of {MIN_BLOCKS} (a convention). "
                "At the estimated block length the null's variance is then expected to be about "
                f"{100 * expected_variance_shortfall(dep.get('n_blocks', 1)):.0f}% too small, so the test tends "
                "to reject too often; the record is short for its dependence (docs/methods.md, Temporal "
                "dependence).")})
        if not dep.get("dependence_decays", True):
            flags.append({"code": "dependence_not_decaying", "message": (
                f"{where}, {what}: the residual products remained significantly "
                "autocorrelated at every lag examined, so the record has no decorrelation lag (for example, "
                "a seasonal cycle). The block length uses the largest significant lag; the null may still "
                "be too narrow.")})
    return flags


def review_flags(result: "PooledResult | HighFlowResult") -> list[dict]:
    """Conditions under which a pathway's result needs a person's review, each
    as {"code", "message"}. Flags never change the selection.

    low_samples_per_dimension : a step had fewer samples per dimension than
        the heuristic threshold (SelectionConfig.min_samples_per_dimension).
    zero_flow : a series has many zero values.
    seasonal_record_too_short : the record is too short to estimate the
        seasonal cycle (the median calendar day has data in fewer than
        preprocessing.seasonal.MIN_YEARS years), so the pathway ran without
        conditioning on the time of year (docs/methods.md, Seasonality).
    untestable : the high-flow event sample has fewer than two usable events,
        or too many dimensions for its number of events for the test to be
        defined, so selection was not run.
    meaningful_effect_undetectable : the data cannot detect contributions as
        small as SelectionConfig.meaningful_r2 with 80% power (docs/methods.md,
        Detectable effect and the meaningful-effect threshold).
    wasp_scale_range_too_short : high-flow pathway with the WASP detector: a
        formulation's scale ceiling leaves too few smoothing scales to
        resolve any events (docs/methods.md, Event detection methods).
    low_ensemble_representation, basis_sensitivity : high-flow pathway only
        (docs/methods.md, Samples).
    dependence_not_decaying : in a step's test or check, the residual
        products' autocorrelation did not become insignificant within the
        lags examined, so no decorrelation lag exists in the record
        (docs/methods.md, Dependence between samples).
    adequacy_check_too_few_samples : an adequacy check that did not reject
        ran on fewer samples than the dependence can be estimated from
        (MIN_ACF_N); such a check can detect little, so the tier was kept
        for lack of evidence against it, not because its assumption was shown
        to hold.
    few_blocks : in a step's test or check, the block resampling rested on
        fewer than MIN_BLOCKS blocks (for event samples, their span divided by
        the dependence lag) (a heuristic: the resampled null's spread
        is then estimated imprecisely and the test tends to reject too often).
    dependent_samples_permuted : in a step's CMI test, the samples exchanged
        by the local permutation were not indistinguishable from independent
        sampling (their decorrelation lag exceeded one sampling step).
    crossfit_blocks_short : in a step's nonlinear test or effect size, the
        cross-fitting blocks were shorter than one dependence lag (record
        spanning fewer than four lags), or a block's fit kept its neighbouring
        blocks for lack of samples, so dependence can leak between fitting and
        prediction.
    low_rank_material : in a step's nonlinear test or effect size, a kernel
        factor reached the rank cap and the approximation may move the effect
        size by as much as the effect-size floor (half-rank check).
    formulation_missing_values : a formulation lacks values at timesteps
        where the observations have them, which removes those timesteps from
        the common basis of every series.
    cmi_calibration_not_checked : the CMI tier made the selection, with a
        non-empty conditioning set, without its calibration check (turned
        off in the configuration); rerun with
        SelectionConfig(cmi_calibration=True) to check it.
    cmi_not_calibrated : the CMI tier was not calibrated on the sample
        (it accepted too many surrogate candidates), so an acceptance or an
        escalation to that tier was withheld (docs/methods.md, Free-form
        tier).
    selected_encompassed : a candidate encompasses a selected formulation:
        given the formulations selected before it, the candidate adds
        information beyond the selected formulation and the selected
        formulation adds no contribution of practical size beyond the
        candidate, either not significant or below the acceptance floor
        (docs/methods.md, Interchangeable formulations).
    escalation_unsupported : an escalation leads to the CMI tier, which is
        not calibrated on its sample, so it was not run: after a power-only
        escalation the current tier's selection stands, possibly incomplete;
        after a validity escalation no selection is reported
        (docs/methods.md, Selection).
    escalated_tier_undetectable : the committed tier was reached by
        escalation, and the round at which it stopped (not accepting its best
        candidate) could not detect a contribution as large as the margin
        (meaningful-effect threshold, else the acceptance floor) with 80%
        power: the stop reflects too few samples, not an absence of
        information.

    event_terms_correlated : in a step on high-flow events in clusters,
        the terms of the statistic were correlated between consecutive
        clusters (docs/methods.md, Dependence between samples).

    The resampling flags (few_blocks, crossfit_blocks_short,
    dependent_samples_permuted, dependence_not_decaying) are raised for every
    step's tests and also for the test that led from each tier to the next,
    which is not part of the committed selection.
    """
    flags = []
    missing = getattr(result, "missing_values", None) or {}
    if missing.get("formulation_missing"):
        removed = {k: v for k, v in missing["excluded_by_formulation"].items() if v > 0}
        flags.append({"code": "formulation_missing_values", "message": (
            "formulation(s) lack values where the observations have them, removing those timesteps from the "
            "analysis of every series: " + ", ".join(f"{k}={v}" for k, v in removed.items()) + ".")})
    tier = result.committed_tier or result.measure
    sel = result.selection
    if (sel is not None and tier == "cmi" and not result.cmi_calibration
            and (isinstance(result, HighFlowResult) or any(st.n_selected_before > 0 for st in sel.trajectory))):
        flags.append({"code": "cmi_calibration_not_checked", "message": (
            "the CMI tier made this selection given previously selected formulations without its calibration "
            "check on surrogate candidates; rerun with SelectionConfig(cmi_calibration=True) to confirm that it "
            "is calibrated on this sample.")})
    if sel is not None:
        for i, step in enumerate(sel.trajectory, start=1):
            cal = step.cmi_calibration
            if cal is not None and not cal["calibrated"]:
                flags.append({"code": "cmi_not_calibrated", "message": (
                    f"round {i} ({step.formulation}): the CMI tier accepted {cal['false_acceptances']} of "
                    f"{cal['n_surrogates']} surrogate candidates carrying no information beyond the conditioning "
                    f"set (binomial p={cal['p_value']:.3g}), so it is not calibrated on this sample; selection "
                    "ended with the formulations it had already accepted.")})
    if sel is not None:
        for rec in sel.equivalence:
            for cmp in rec["comparisons"]:
                if (cmp["relation"] == "encompasses_selected"
                        and cmp["candidate_adds_beyond_selected"]["p_value"] < rec["flag_level"]):
                    fwd, rev = cmp["candidate_adds_beyond_selected"], cmp["selected_adds_beyond_candidate"]
                    alpha, floor = rec.get("alpha", 0.05), rec.get("effect_size_floor")
                    if rev["p_value"] >= alpha or floor is None:
                        why = f"not significant (p={rev['p_value']:.3g})"
                    else:
                        why = (f"significant (p={rev['p_value']:.3g}), but its estimated size "
                               f"({rev.get('effect', float('nan')):.3g}) is below the acceptance floor ({floor:.3g})")
                    flags.append({"code": "selected_encompassed", "message": (
                        f"round {rec['round']}: '{cmp['candidate']}' adds information beyond the selected "
                        f"'{rec['selected']}' (p={fwd['p_value']:.3g}, effect {fwd.get('effect', float('nan')):.3g}; significant at "
                        f"{rec['flag_level']:.3g}, Bonferroni across the round's comparisons), and "
                        f"'{rec['selected']}' adds no contribution of practical size beyond it: {why}. The candidate "
                        "may be the better choice at that round.")})
    attempts = getattr(result, "attempted_tiers", None) or []
    provisional = "; ".join(f"{a['tier']} tier: {a['provisional_selection']}" for a in attempts
                            if a.get("provisional_selection")) or "none"
    for attempt in attempts:
        if not attempt.get("run", True):
            extra = ("" if not attempt.get("committed") else
                     f" Selections made before the assumptions were rejected, for information only: {provisional}.")
            flags.append({"code": "escalation_unsupported", "message": f"{attempt['reason']}.{extra}"})
    own_steps = [st for st in (sel.trajectory if sel is not None else []) if st.tier in (None, tier)]
    margin = (result.meaningful_r2 if result.meaningful_r2 is not None
              else sel.equivalence[0].get("margin") if sel is not None and sel.equivalence and sel.equivalence[0].get("margin")
              else getattr(result, "adequacy_floor", None) or DEFAULT_ADEQUACY_FLOOR)
    stop = own_steps[-1] if own_steps and not own_steps[-1].accepted else None
    if (len(attempts) > 1 and attempts[-1].get("run", True) and stop is not None
            and stop.detectable_r2 is not None and stop.detectable_r2 > margin):
        flags.append({"code": "escalated_tier_undetectable", "message": (
            f"selection stopped in the {tier} tier, reached by escalation, at a round that could not detect a "
            f"contribution of the size that matters on its sample: its smallest detectable R^2 is "
            f"{format_share(stop.detectable_r2)}, above the margin of {format_share(margin)} (the meaningful-effect "
            f"threshold when set, otherwise the acceptance floor). The stop then reflects too few samples, not an "
            f"absence of information. Selections discarded by a validity escalation, for information only: "
            f"{provisional}.")})
    if sel is not None:
        for i, step in enumerate(sel.trajectory, start=1):
            if step.low_samples_per_dimension:
                flags.append({"code": "low_samples_per_dimension", "message": (
                    f"round {i} ({step.formulation}): {step.samples_per_dimension:.1f} samples per dimension "
                    "(candidate, observations and selected set), below the heuristic threshold. Power is low in "
                    "every tier, and the nonlinear and CMI tiers may accept too often, because their regressions "
                    "or neighbourhoods in the selected set are then poorly estimated. The threshold is an analogy "
                    "to events-per-variable guidance in regression, not a derived requirement.")})
    if sel is not None:
        for i, step in enumerate(sel.trajectory, start=1):
            sources = [(f"{step.tier or tier} tier test", step.dependence)] + [
                (a.get("check", "check"), a.get("dependence")) for a in step.adequacy]
            for a in step.adequacy:
                dep = a.get("dependence") or {}
                if a.get("adequate") and dep.get("too_few_to_assess"):
                    flags.append({"code": "adequacy_check_too_few_samples", "message": (
                        f"round {i} ({step.formulation}), {a.get('check', 'check')}: the check ran on "
                        f"{dep.get('n_samples')} samples, fewer than the {MIN_ACF_N} from which their dependence can "
                        "be estimated, and can detect little; the tier was kept for lack of evidence against it, "
                        "not because its assumption was shown to hold. On long records the samples are spaced at the "
                        "dependence lag, so a long lag leaves few of them.")})
            flags += _dependence_flags(f"round {i} ({step.formulation})", sources)
    for attempt in attempts:
        test = attempt.get("escalation_test")
        if test and test.get("dependence"):
            flags += _dependence_flags(
                f"{attempt['tier']} tier, round {test['round']} ({test['candidate']})",
                [(f"{test['check']}, the test that led to the next tier"
                  + (f" (p={test['p_value']:.3g})" if test.get("p_value") is not None else ""),
                  test["dependence"])])
    if result.zero_flow_flag:
        flags.append({"code": "zero_flow", "message": "at least one series has a high fraction of zero flow."})
    if (getattr(result, "seasonal_conditioning", None) or {}).get("record_too_short"):
        flags.append({"code": "seasonal_record_too_short", "message": (
            "the record is too short to estimate the seasonal cycle (the median calendar day has data in fewer "
            "than three years), so the series were analysed with their seasonal cycles: a formulation can be "
            "credited with reproducing the seasonal cycle.")})
    if isinstance(result, PooledResult) and result.meaningful_r2 is not None and sel is not None \
            and sel.trajectory and sel.trajectory[0].detectable_r2 is not None \
            and sel.trajectory[0].detectable_r2 > result.meaningful_r2:
        detectable = sel.trajectory[0].detectable_r2
        flags.append({"code": "meaningful_effect_undetectable", "message": (
            f"the smallest contribution the record can detect with 80% power in round 1 (R^2 = {format_share(detectable)}) "
            f"exceeds the meaningful threshold (R^2 = {format_share(result.meaningful_r2)}): formulations contributing at "
            "the meaningful level would usually go unselected, and the effect sizes of accepted formulations "
            "are likely overestimated.")})
    if isinstance(result, HighFlowResult):
        if result.power_stop_reason:
            flags.append({"code": "untestable", "message": (
                f"high-flow selection not run: {result.power_stop_reason}")})
        elif (result.meaningful_r2 is not None and result.detectable_r2 is not None
              and result.detectable_r2 > result.meaningful_r2):
            flags.append({"code": "meaningful_effect_undetectable", "message": (
                f"the smallest contribution these events can detect with 80% power (R^2 = {format_share(result.detectable_r2)}) "
                f"exceeds the meaningful threshold (R^2 = {format_share(result.meaningful_r2)}): formulations contributing at "
                "the meaningful level would usually go unselected, and the effect sizes of accepted formulations "
                "are likely overestimated.")})
        unresolved = sorted(name for name, meta in (result.wasp_metadata or {}).items()
                            if name != OBS_KEY and meta.get("scale_range_too_short"))
        if unresolved:
            flags.append({"code": "wasp_scale_range_too_short", "message": (
                f"WASP could not resolve events in {', '.join(unresolved)}: the series' scale ceiling (its "
                "autocorrelation e-folding time) leaves fewer than two smoothing scales, so no maximum can be "
                "linked across scales. These series contribute no events or peaks to the consensus events.")})
        if result.ensemble_representation.get("low_confidence"):
            flags.append({"code": "low_ensemble_representation", "message": (
                "observed event shapes are poorly reconstructed by the formulations' basis; the ensemble "
                "may not represent observed behaviour.")})
        if result.requires_closer_inspection:
            flags.append({"code": "basis_sensitivity", "message": (
                "repeating the stopping round on a basis fitted to a remaining candidate's own curves accepted "
                "a candidate (Bonferroni-adjusted across bases); the decision may depend on the FPCA basis "
                "(see basis_sensitivity_check).")})
    return flags


def _review_header(result) -> dict:
    flags = review_flags(result)
    return {"review_required": bool(flags), "review_flags": flags}


def selection_to_records(sel: SelectionResult | None, n_features: int = 1) -> dict:
    keys = ("selected", "trajectory", "unconditional_association", "unconditional_effect", "stopped_reason", "ill_posed",
            "ill_posed_reason", "equivalence", "equivalent_sets", "equivalent_sets_summary", "escalated_from")
    if sel is None:
        return {k: None for k in keys}
    return {
        "selected": sel.selected,
        "trajectory": [
            {
                "formulation": st.formulation,
                "tier": st.tier,
                "association": st.association,
                "effect": st.effect,
                "p_value": st.p_value, "accepted": st.accepted,
                "magnitude_unsubstantiated": st.magnitude_unsubstantiated,
                "n_selected_before": st.n_selected_before,
                "n_candidates_tested": st.n_candidates_tested,
                "dependence_lag": st.dependence_lag,
                "block_length": st.block_length,
                "dependence": st.dependence,
                "null_mean": st.null_mean, "null_std": st.null_std,
                "diagnostics": st.diagnostics,
                "adequacy": st.adequacy,
                "other_candidates_checked": st.other_candidates_checked,
                "cmi_calibration": st.cmi_calibration,
                "samples_per_dimension": st.samples_per_dimension,
                "low_samples_per_dimension": st.low_samples_per_dimension,
                "detectable_r2": st.detectable_r2,
                "effective_sample_size": st.effective_sample_size,
                "detectability": (describe_detectable(st.detectable_r2, n_features)
                                  if st.detectable_r2 is not None else None),
            }
            for st in sel.trajectory
        ],
        "unconditional_association": sel.unconditional_association,
        "unconditional_effect": sel.unconditional_effect,
        "stopped_reason": sel.stopped_reason,
        "ill_posed": sel.ill_posed,
        "ill_posed_reason": sel.ill_posed_reason,
        "equivalence": sel.equivalence,
        "equivalent_sets": sel.equivalent_sets,
        "equivalent_sets_summary": sel.equivalent_sets_summary,
        "escalated_from": sel.escalated_from,
    }


def pooled_result_to_dict(result: PooledResult) -> dict:
    """JSON-ready dict of a PooledResult (an explicit allowlist of fields)."""
    sel = result.selection
    d = {
        **_review_header(result),
        "gage": result.gage,
        "n_timesteps": result.n_timesteps,
        "n_complete_timesteps": result.n_complete_timesteps,
        "missing_values": result.missing_values,
        "zero_flow_fraction": result.zero_flow_fraction,
        "zero_flow_flag": result.zero_flow_flag,
        "pearson_correlation": result.pearson_correlation,
        "pearson_correlation_transformed": result.pearson_correlation_transformed,
        "min_temporal_lag": result.min_temporal_lag,
        "measure": result.measure,
        "dependence_aware_subsampling_used": result.dependence_aware_subsampling_used,
        "n_after_subsampling": result.n_after_subsampling,
        "natural_independent_sample_size": result.natural_independent_sample_size,
        "notes": result.notes,
        "committed_tier": result.committed_tier,
        "attempted_tiers": result.attempted_tiers,
        "adequacy_floor": result.adequacy_floor,
        "seed": result.seed,
        "cmi_calibration": result.cmi_calibration,
        "meaningful_r2": result.meaningful_r2,
        "seasonal_conditioning": result.seasonal_conditioning,
        "ill_posed": sel.ill_posed,
        "ill_posed_reason": sel.ill_posed_reason,
        "detectability": (describe_detectable(sel.trajectory[0].detectable_r2)
                          if sel.trajectory and sel.trajectory[0].detectable_r2 is not None else None),
    }
    d.update({f"pooled_{k}": v for k, v in selection_to_records(sel).items()})
    return d


def _events_to_records(events: list[ExpandedEvent]) -> list[dict]:
    return [
        {
            "event_id": ev.event_id,
            "start": ev.start,
            "end": ev.end,
            "duration_timesteps": ev.duration_timesteps,
            "source_consensus_event_ids": ev.source_consensus_event_ids,
            "peak_index": ev.peak_index,
            "start_time": ev.start_time.isoformat() if ev.start_time is not None else None,
            "end_time": ev.end_time.isoformat() if ev.end_time is not None else None,
            "peak_time": ev.peak_time.isoformat() if ev.peak_time is not None else None,
        }
        for ev in events
    ]


def high_flow_result_to_dict(result: HighFlowResult) -> dict:
    """JSON-ready dict of a HighFlowResult (an explicit allowlist of fields)."""
    sel = result.selection
    d = {
        **_review_header(result),
        "gage": result.gage,
        "n_timesteps": result.n_timesteps,
        "missing_values": result.missing_values,
        "pearson_correlation": result.pearson_correlation,
        "zero_flow_fraction": result.zero_flow_fraction,
        "zero_flow_flag": result.zero_flow_flag,
        "min_temporal_lag": result.min_temporal_lag,
        "event_dependence": result.event_dependence,
        "seasonal_conditioning": result.seasonal_conditioning,
        "pot_ratio": result.pot_ratio,
        "recession_ratio": result.recession_ratio,
        "wasp_metadata": result.wasp_metadata,
        "n_raw_consensus_events": result.n_raw_consensus_events,
        "n_components": result.n_components,
        "fpca_fit_source": result.fpca_fit_source,
        "fpca_cumulative_explained_variance": result.fpca_cumulative_explained_variance,
        "fpca_component_orthogonality_error": result.fpca_component_orthogonality_error,
        "fpca_regularization_lambda": result.fpca_regularization_lambda,
        "fpca_n_grid": result.fpca_n_grid,
        "fpca_explained_variance_ratio": result.fpca_explained_variance_ratio,
        "fpca_reconstruction_fidelity": result.fpca_reconstruction_fidelity,
        "peak_phase_spread": result.peak_phase_spread,
        "requires_closer_inspection": result.requires_closer_inspection,
        "detectable_r2": result.detectable_r2,
        "detectability": (describe_detectable(result.detectable_r2, result.n_components)
                          if result.detectable_r2 is not None else None),
        "meaningful_r2": result.meaningful_r2,
        "power_stop_reason": result.power_stop_reason,
        "ensemble_representation": result.ensemble_representation,
        "high_flow_power": result.high_flow_power,
        "measure": result.measure,
        "committed_tier": result.committed_tier,
        "attempted_tiers": result.attempted_tiers,
        "adequacy_floor": result.adequacy_floor,
        "seed": result.seed,
        "cmi_calibration": result.cmi_calibration,
        "ill_posed": sel.ill_posed if sel is not None else False,
        "ill_posed_reason": sel.ill_posed_reason if sel is not None else None,
        "notes": result.notes,
    }
    d.update({f"high_flow_{k}": v for k, v in selection_to_records(sel, result.n_components).items()})
    d["basis_sensitivity_check"] = result.basis_sensitivity_check
    d["reconstruction_fidelity_by_event"] = result.reconstruction_fidelity_by_event
    d["high_flow_events"] = _events_to_records(result.high_flow_events)
    return d


def basin_result_to_dict(result: BasinResult) -> dict:
    flags = [{"pathway": "pooled", **f} for f in review_flags(result.pooled)]
    if result.high_flow is not None:
        flags += [{"pathway": "high_flow", **f} for f in review_flags(result.high_flow)]
    return {
        "review_required": bool(flags),
        "review_flags": flags,
        "gage": result.gage,
        "zero_flow_flag": result.zero_flow_flag,
        "zero_flow_fraction": result.zero_flow_fraction,
        "pooled": pooled_result_to_dict(result.pooled),
        "high_flow": high_flow_result_to_dict(result.high_flow) if result.high_flow is not None else None,
    }


def _pathway_outcome(label: str, result) -> str:
    """One clause: the selected set, whether any selected formulation could
    be replaced, and the largest further contribution that would go
    undetected."""
    sel = result.selection if result is not None else None
    if sel is None:
        reason = getattr(result, "power_stop_reason", None) if result is not None else None
        return f"{label}: not run" + (f" ({reason})" if reason else "")
    selected = [st.formulation for st in sel.trajectory if st.accepted]
    if not selected:
        return f"{label}: nothing selected"
    text = f"{label} {{{', '.join(selected)}}}"
    if sel.equivalence:
        swaps, better = [], []
        for rec in sel.equivalence:
            for c in rec["comparisons"]:
                if c["relation"] in ("equivalent", "not_distinguishable"):
                    kind = "equivalent" if c["relation"] == "equivalent" else "not distinguishable"
                    swaps.append(f"{rec['selected']} by {c['candidate']} ({kind})")
                elif c["relation"] == "encompasses_selected":
                    better.append(f"{c['candidate']} may be better than {rec['selected']}")
        text += ": " + ("replaceable " + "; ".join(swaps) if swaps else "none replaceable")
        if better:
            text += "; " + "; ".join(better)
    last = sel.trajectory[-1]
    if not last.accepted and last.detectable_r2 is not None:
        text += f"; further contributions below R^2 = {format_share(last.detectable_r2)} undetectable"
        prior = sel.escalated_from
        if prior and prior.get("detectable_r2") is not None:
            text += f" (below {format_share(prior['detectable_r2'])} in the {prior['tier']} tier, before escalation)"
    return text


def outcome_line(pooled: "PooledResult | None" = None, high_flow: "HighFlowResult | None" = None) -> str:
    """The outcome of a basin's selection in one line: for each pathway run,
    the selected set, whether any selected formulation could be replaced by
    another (docs/methods.md, Interchangeable formulations), and the largest
    further contribution that would go undetected; with both pathways, the
    union of their selected sets."""
    gage = (pooled or high_flow).gage
    parts = []
    if pooled is not None:
        parts.append(_pathway_outcome("pooled", pooled))
    if high_flow is not None:
        parts.append(_pathway_outcome("high-flow", high_flow))
    line = f"{gage}: " + " | ".join(parts)
    if pooled is not None and high_flow is not None:
        union = []
        for res in (pooled, high_flow):
            for st in (res.selection.trajectory if res.selection is not None else []):
                if st.accepted and st.formulation not in union:
                    union.append(st.formulation)
        line += f" | both pathways: {{{', '.join(union)}}}"
    return line

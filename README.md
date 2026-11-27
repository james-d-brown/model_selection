# model_selection

Per-basin selection of hydrologic model formulations by the information each contributes about observed streamflow beyond the formulations already selected.

The methods are described in full in [docs/methods.md](docs/methods.md), which is the canonical description. This README summarizes it, and its links lead to the relevant sections of the methods.

## Overview

This pipeline is concerned with model selection. It identifies, for one basin, the model formulations whose simulated flow carries information about observed flow that the formulations already selected do not. Conditional independence of a candidate and the observations, given the selected formulations, is the formal statement that the candidate contributes no such information, and is the hypothesis each selection step tests.

Evaluating each formulation separately against observations answers "how well does this model reproduce the observations?". Formulations that share forcing, structure and other attributes are closely related, so several can score well while carrying largely the same information. Model selection asks a different question: what information does a candidate formulation contribute about the observations that the selected formulations do not? A formulation whose information is already carried by the selected set adds nothing, however well it scores on its own. Selection avoids retaining formulations that are redundant with one another.

### Samples

Each sample is one time step over the whole record, so a formulation's added information is measured across all flow conditions. Every test conditions on the time of year, so a formulation is not credited with simple climatological information (see [Seasonality](docs/methods.md#seasonality)). Samples are strongly statistically dependent in time.

The candidate, the observations and the selected formulations are compared on matched samples: the same time step. Better timing of the response, through better representation of routing, storage or channel hydraulics, therefore counts as information, as a difference in flow at the time it occurs.

### The procedure

The procedure has five steps:

1. **Samples.** Every series is transformed to a common scale, and the samples are formed: one per time step (see [Samples](docs/methods.md#samples)).

2. **Selection.** Formulations are added one at a time by greedy forward selection. In each round, the formulation not yet selected (a *candidate*) that carries the most information about the observations beyond the selected formulations is added if it meets two conditions. First, the measured information must be statistically significant. Second, it must reach the acceptance floor, an effect-size floor set as a minimum share of the observations' variation that is not yet explained (see [Acceptance test](docs/methods.md#acceptance-test)). Selection stops at the first round whose best candidate does not meet both conditions. Selection runs through three tiers of tests as a cascade, from the least general and most powerful to the most general and least powerful, moving to a more general tier only when an escalation test finds the simpler tier's assumptions violated (see [Selection](docs/methods.md#selection)).

3. **Hypothesis tests.** Each candidate is tested against the null hypothesis that it carries no information about the observations beyond the selected formulations (see [Hypothesis tests](docs/methods.md#hypothesis-tests)). No single test is both valid and powerful for every form of dependence, so the three tiers remove the dependence on the selected formulations under different assumptions, and each detects a different range of remaining dependence between the candidate and the observations. The tests account for statistical dependence between samples (see [Dependence between samples](docs/methods.md#dependence-between-samples)).

    - **Linear tier.** Assumes the selected formulations act linearly on the candidate and the observations, and detects a linear relationship between the candidate and the observations that the selected formulations do not account for, averaged over states. It does not model effect modification of any type: a relationship that varies across states is detected only through its average, which cancels when the relationship changes direction. Uses linear regression (partial correlation).

    - **Nonlinear tier.** Allows the selected formulations to act nonlinearly, and detects dependence of any shape between the candidate and the observations that the selected formulations do not account for, including dependence that varies with state. It does not model effect modification either, but detects it through many transformations of the variables, so it misses effect modification only when it cancels exactly across states. A candidate is accepted only if it also explains enough of the observations' variation, through a function of the candidate; information about the spread or shape of the observations alone is not enough (see [Assumptions and limitations](docs/methods.md#assumptions-and-limitations)). Uses nonlinear regression.

    - **Free-form tier.** Assumes only that the candidate's distribution changes smoothly with the selected formulations, and detects dependence of any shape within states, including effect modification that cancels across states, to any degree, which the other tiers can miss. Uses conditional mutual information.

4. **Post-selection checks.** The formulation chosen in a round may be one of several that carry the same information, so each selected formulation is compared with the other candidates, and the sets of formulations the data support equally are identified. The checks are reported, but do not affect the selection (see [Post-selection checks](docs/methods.md#post-selection-checks)).

5. **Results.** Every result reports the selection, the checks, the smallest contribution the data could detect, and the diagnostics of the assumptions behind each decision (see [Results](docs/methods.md#results)).

Every default setting, its source and the configuration that changes it are listed in the methods, [Appendix A](docs/methods.md#appendix-a-defaults-and-configuration). The assumptions behind each step, and their limitations, are listed in the methods, [Assumptions and limitations](docs/methods.md#assumptions-and-limitations).

## Installation

Python 3.10 or later. From a clone of the repository:

```
pip install -e .
```

Observations are read from the USGS Water Data APIs. Set the environment variable `API_USGS_PAT` to your USGS API token.

## Usage

```python
from model_selection import load_basin, run_pooled_analysis, Censoring, SelectionConfig, OutputConfig

basin_dir = "path/to/basin"        # folder with the formulations' Parquet output
gage = "02198690"                   # USGS site number
start_date, end_date = "2015-10-01", "2020-09-30"

basin = load_basin(basin_dir, gage, start_date, end_date)   # USGS token from API_USGS_PAT
result = run_pooled_analysis(basin, SelectionConfig(measure="cascade"), output=OutputConfig(output_dir="out/"))
censored = run_pooled_analysis(basin, SelectionConfig(measure="cascade"), censoring=Censoring())   # high flows
print(result.committed_tier, result.selection.selected)
```

`SelectionConfig.measure` is `"cascade"` (the default), or `"linear"`, `"nonlinear"` or `"cmi"` to use one tier throughout. Settings are grouped into frozen dataclasses, each owned by the component it configures: `SelectionConfig` and `AdequacyConfig` (selection), `Censoring` (censoring at high flow) and `OutputConfig` (pipeline). All fields have defaults, documented on each class. `run_pooled_analysis` and `run_basin` take these objects. `run_pooled_analysis(..., censoring=Censoring())` runs the analysis with the candidates and observations censored at the high-flow quantile of their own record, 0.90 by default (docs/methods.md, Censoring at high flow), and writes `{gage}_pooled_censored_result.json`; `run_basin(..., censoring=...)` runs it alongside the uncensored analysis.

The experimental high-flow event pathway (docs/methods.md, Appendix E: Experimental high-flow event pathway) is in `model_selection.experimental`, with its settings: `EventConfig` with `POTConfig` and `WASPConfig` (event detection) and `FPCAConfig` (high-flow event sample). It runs on its own as `model_selection.experimental.run_high_flow_analysis`, or with the pooled pathway as `run_basin(..., run_high_flow=True)`, and issues an `ExperimentalWarning` once per process.

Each result's JSON output begins with `review_required` and `review_flags`, which list every condition that needs a person's review, such as a step with few samples per dimension; flags are also logged as warnings and never change the selection. The output then records, for every selection step, the winning formulation, its statistic and the effect size its acceptance used, its p-value and the assumption diagnostics. To review many basins at once:

```
python scripts/summarize_results.py <results directory> summary.csv
```

lists every result in one table, those requiring review first.

Event detection on a single series (experimental), for example a USGS record without model output:

```python
from model_selection.experimental.event_detection import detect_events_for_series, EventConfig
series = ...  # a pandas Series of streamflow with a DatetimeIndex
events = detect_events_for_series(series, EventConfig(method="pot")).decluster_result
```

New detection methods implement `experimental.event_detection.base.EventDetector` and can be passed as `detect_events_for_series(series, config, detector=...)`.

## Layout

```
model_selection/
├── model_selection/       the package
│   ├── io/                model output, USGS observations, JSON output
│   ├── preprocessing/     transforms, conditioning on the time of year, gap regularization
│   ├── statistics/
│   │   ├── conditional_independence/  the three tiers' tests
│   │   ├── adequacy/      the cascade's adequacy (specification and interaction) tests
│   │   ├── regression/    least squares, cross-fitted kernel ridge residuals of kernel feature maps
│   │   ├── association/   nearest-neighbour information estimators (mixed data), canonical correlations, distance helpers
│   │   └── significance/  block lengths, resampling, p-values, bias correction
│   ├── selection/         tiers, greedy selection, the cascade, configuration
│   ├── pipeline/
│   │   ├── pooled/        the pooled pathway
│   │   ├── shared.py      steps shared with the experimental pathway
│   │   ├── results.py     result types and their JSON form
│   │   └── orchestration.py   run_basin: the pathways for one basin
│   ├── visualization/     Plotly figures and HTML writers
│   └── experimental/      the high-flow event pathway (high_flow/), event detection (event_detection/: pot/, wasp/,
│                          consensus events, windows), fpca/, hydrology/ (WASP scale ceiling), visualization/
├── scripts/               multi-basin summary table; HTML rendering of the docs and Figure 1; diagnostics/ (known-answer tests)
├── tests/                 mirrors the package layout
├── docs/methods.md        methods, assumptions and references
└── LICENSE                Apache License, Version 2.0
```

Dependencies point one way: `statistics` knows nothing about hydrology or the pipeline; `selection` builds on `statistics`; `pipeline` combines them; `visualization` is used by the pipeline only for writing output; `experimental` builds on all of them, and nothing outside it imports it except `run_basin(..., run_high_flow=True)`.

## Tests

```
pip install -e ".[test]"
pytest
```

The tests also run on GitHub for every push and pull request (`.github/workflows/tests.yml`).

## Documentation in HTML

GitHub displays `docs/methods.md` with its equations. To read the documentation offline in a browser, run

```
pip install -e ".[docs]"
python scripts/render_docs.py
```

which writes `README.html` and `docs/methods.html` from the Markdown files. The Markdown files remain the source, and the HTML files are not tracked.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

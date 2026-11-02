# Diagnostics

`temporal_calibration.py` needs no data: it simulates autocorrelated series
satisfying each test's null hypothesis, builds samples as the pooled pathway
does, and reports each test's false-positive rate (docs/methods.md, Temporal
dependence). Run it from the project root with
`python -m scripts.diagnostics.temporal_calibration`.

`event_merging.py` reports, for one loaded basin, how the high-flow
pathway's consensus events are built (high-flow runs, joins by at least half of
the formulations' own events, widened windows separated at overlaps),
without running selection.

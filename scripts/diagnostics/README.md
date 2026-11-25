# Diagnostics

`known_answer_check.py` runs the pooled pathway on synthetic daily basins
whose correct selection is known, and reports, for each replicate, the
forward selection, the final set after the backward pass, the formulations
the backward pass removed, the class of every formulation, the number of
post-selection tests, and the time taken by each stage. It needs no data.

Cases (`--case`):

- `simple`: five formulations; the correct set is the exact baseflow and
  quickflow; when a noisy composite of the two is selected first, the
  backward pass removes it.
- `many`: eight formulations (three exact components, three noisy
  composites, a near-copy of one component, and noise); the correct set is
  the three components, with the near-copy exchangeable.
- `many_censored`: the same roles, built so that the correct set is the same
  with and without censoring at a high quantile.

Run it from the project root, for example

```
python -m scripts.diagnostics.known_answer_check --case many --reps 3 --workers 3
python -m scripts.diagnostics.known_answer_check --case many_censored --reps 3 --workers 3 --censored
python -m scripts.diagnostics.known_answer_check --case many --reps 1 --plot-only
```

Options: `--censored` (with `--quantile`, default 0.9) runs the censored
pooled pathway; `--plot` and `--plot-only` write each replicate's series to an
HTML page; `--profile` runs each replicate under Python's profiler;
`--quiet` turns off the stage-by-stage progress; `--years`, `--measure`,
`--n-perm`, `--first-seed` and `--workers` set the record length, the tier
or cascade, the number of resamples, the seeds and the parallel replicates.
Each replicate runs single-threaded. The script's docstring describes each
case's construction and what counts as a correct outcome.

The other diagnostic scripts used in developing the methods are kept locally
and are not part of the repository.

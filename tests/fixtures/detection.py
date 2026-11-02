"""Per-model detection on plain arrays for tests (hourly index assumed)."""
import pandas as pd

from model_selection.event_detection import EventConfig, POTConfig, detect_events_per_series


def own_model_windows(raw_formulations, quantile_threshold=0.90, pot_ratio=2.0 / 3.0, recession_ratio=0.25,
                      min_peak_separation=1, method="pot", **_ignored):
    """{name: DeclusterResult} for arrays, using detect_events_per_series."""
    series = {name: pd.Series(v, index=pd.date_range("2000-01-01", periods=len(v), freq="h", tz="UTC"))
              for name, v in raw_formulations.items()}
    config = EventConfig(method=method, high_flow_quantile=quantile_threshold,
                         pot=POTConfig(pot_ratio=pot_ratio, recession_ratio=recession_ratio,
                                       min_peak_separation=min_peak_separation))
    return {name: r.decluster_result for name, r in detect_events_per_series(series, config).items()}

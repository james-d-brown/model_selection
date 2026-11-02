import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from model_selection.io.basin import BasinData, load_basin
from model_selection.io.units import convert_streamflow_units
from model_selection.io.usgs import pull_usgs_obs


class TestHydroPipelineIO(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.test_dir = tempfile.mkdtemp()
        self.basin_dir = os.path.join(self.test_dir, "basin_data")
        os.mkdir(self.basin_dir)

        self.create_mock_parquet(
            path=os.path.join(self.basin_dir, "model_topmodel.parquet"),
            feature_id="02198690",
            formulation="noahowp_topmodel",
            unit="cm/s",
            start_str="2023-10-01 00:00:00",
            end_str="2023-10-01 04:00:00"
        )

        self.create_mock_parquet(
            path=os.path.join(self.basin_dir, "model_nwm.parquet"),
            feature_id="02198690",
            formulation="nwm_v3",
            unit="cm/s",
            start_str="2023-10-01 00:00:00",
            end_str="2023-10-01 04:00:00"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def create_mock_parquet(self, path: str, feature_id: str, formulation: str, unit: str, start_str: str, end_str: str):
        times = pd.date_range(start=start_str, end=end_str, freq="1h")
        df = pd.DataFrame({
            "sim_flow": np.linspace(10.0, 50.0, num=len(times)).astype(np.float32)
        }, index=pd.Index(times, name="value_time"))

        table = pa.Table.from_pandas(df)

        custom_metadata = {
            b"feature_id": feature_id.encode("utf-8"),
            b"formulation": formulation.encode("utf-8"),
            b"variable_name": b"streamflow",
            b"measurement_unit": unit.encode("utf-8"),
            b"val_start": start_str.encode("utf-8"),
            b"val_end": end_str.encode("utf-8"),
        }

        existing_meta = table.schema.metadata or {}
        combined_meta = {**existing_meta, **custom_metadata}
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(table, path)

    def test_unit_conversion_math(self):
        sample_data = pd.Series([100.0])
        converted = convert_streamflow_units(sample_data, from_unit="cfs", to_unit="cm/s")
        # FIX: Extract direct scalar value instead of passing internal index proxy
        self.assertTrue(np.isclose(converted.iloc[0], 2.8316846592))

        restored = convert_streamflow_units(converted, from_unit="cm/s", to_unit="cfs")
        self.assertTrue(np.isclose(restored.iloc[0], 100.0))

    @patch.dict(os.environ, {"API_USGS_PAT": "valid_test_token"})
    def test_pull_usgs_obs_handles_real_wdfn_schema(self):
        mock_waterdata = MagicMock()

        example_api_df = pd.DataFrame({
            "time": ["2023-10-01T00:00:00Z", "2023-10-01T00:15:00Z"],
            "value": [150.0, 155.0],
            "time_series_id": ["65e9e1b16ade44ac95691436de0ea679", "65e9e1b16ade44ac95691436de0ea679"]
        })

        mock_waterdata.get_continuous.return_value = (example_api_df, {"mock_meta": "info"})

        with patch.dict(sys.modules, {
            "dataretrieval": MagicMock(waterdata=mock_waterdata),
            "dataretrieval.waterdata": mock_waterdata
        }):
            result_series = pull_usgs_obs(
                gage="02198690",
                start_date="2023-10-01T00:00:00Z",
                end_date="2023-10-02T00:00:00Z"
            )

            self.assertEqual(len(result_series), 2)
            self.assertIsInstance(result_series.index, pd.DatetimeIndex)
            self.assertEqual(result_series.iloc[0], 150.0)
            self.assertEqual(result_series.index[0], pd.to_datetime("2023-10-01 00:00:00", utc=True))

    @patch.dict(os.environ, {"API_USGS_PAT": "valid_test_token"})
    def test_pull_usgs_obs_automatically_chunks_large_time_windows(self):
        mock_waterdata = MagicMock()
        captured_time_strings = []

        def get_continuous_side_effect(monitoring_location_id, parameter_code, time):
            captured_time_strings.append(time)

            start_str, end_str = time.split('/')
            self.assertTrue(start_str.endswith("Z"), f"Start chunk '{start_str}' is missing 'Z'")
            self.assertTrue(end_str.endswith("Z"), f"End chunk '{end_str}' is missing 'Z'")

            dummy_df = pd.DataFrame({
                "time": [start_str],
                "value": [50.0],
                "time_series_id": ["65e9e1b16ade44ac95691436de0ea679"]
            })
            return (dummy_df, None)

        mock_waterdata.get_continuous.side_effect = get_continuous_side_effect

        with patch.dict(sys.modules, {
            "dataretrieval": MagicMock(waterdata=mock_waterdata),
            "dataretrieval.waterdata": mock_waterdata
        }):
            result_series = pull_usgs_obs(
                gage="02198690",
                start_date="2010-01-01T00:00:00Z",
                end_date="2015-01-01T00:00:00Z"
            )

            self.assertEqual(mock_waterdata.get_continuous.call_count, 2)
            self.assertEqual(len(result_series), 2)
            self.assertEqual(captured_time_strings[0].split('/')[0], "2010-01-01T00:00:00Z")
            self.assertEqual(captured_time_strings[-1].split('/')[1], "2015-01-01T00:00:00Z")

    @patch.dict(os.environ, {"API_USGS_PAT": "valid_test_token"})
    def test_end_to_end_pipeline_ingestion(self):
        mock_waterdata = MagicMock()

        obs_times = ["2023-10-01T00:00:00Z", "2023-10-01T01:00:00Z", "2023-10-01T02:00:00Z", "2023-10-01T03:00:00Z"]
        mock_obs_df = pd.DataFrame({
            "time": obs_times,
            "value": [100.0, 100.0, 100.0, 100.0],
            "time_series_id": ["65e9e1b16ade44ac95691436de0ea679"] * 4
        })

        mock_waterdata.get_continuous.return_value = (mock_obs_df, None)

        with patch.dict(sys.modules, {
            "dataretrieval": MagicMock(waterdata=mock_waterdata),
            "dataretrieval.waterdata": mock_waterdata
        }):
            result = load_basin(
                basin_dir=self.basin_dir,
                gage="02198690",
                start_date="2023-10-01T00:00:00Z",
                end_date="2023-10-02T00:00:00Z",
                resample_rule="mean"
            )

            self.assertIsInstance(result, BasinData)
            self.assertEqual(result.gage, "02198690")
            self.assertIn("noahowp_topmodel", result.formulations)
            self.assertIn("nwm_v3", result.formulations)
            self.assertTrue(np.isclose(result.obs.iloc[0], 2.8316846592))


if __name__ == "__main__":
    unittest.main()
#
# Copyright 2022 QuantRocket LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import TestCase
from unittest import skipIf
from unittest.mock import patch
import pandas as pd
try:
    from zipline.pipeline import Pipeline, master, EquityPricing
    zipline_installed = True
except ImportError:
    zipline_installed = False
try:
    from quantrocket.utils import segmented_date_range
    quantrocket_installed = True
except ImportError:
    quantrocket_installed = False
from ..pipeline import SegmentedAnalysisProgressMeter, from_pipeline


class FromPipelineTestCase(TestCase):

    @skipIf(not zipline_installed, "zipline not installed")
    @patch("alphalens.pipeline.get_forward_returns")
    def test_initial_universe(self, mock_get_forward_returns):
        """
        Test that the initial universe is passed from the Pipeline to
        get_forward_returns.
        """
        initial_universe = master.SecuritiesMaster.Symbol.latest.eq("AAPL")
        pipeline = Pipeline(
            columns={"close": EquityPricing.close.latest},
            initial_universe=initial_universe)

        idx = index=pd.MultiIndex.from_product(
            [pd.date_range("2018-01-01", "2018-01-02"), ["AAPL"]], names=["date", "asset"])

        def mock_run_pipeline(*args, **kwargs):
            return pd.DataFrame({"close": [1,2]}, index=idx)

        forward_returns = pd.DataFrame({"1D": 1}, index=idx)
        mock_get_forward_returns.return_value = forward_returns

        with patch("alphalens.pipeline.run_pipeline", mock_run_pipeline):

            from_pipeline(pipeline, "2018-01-01", "2018-01-02", factor="close", quantiles=1)

        self.assertEqual(len(mock_get_forward_returns.mock_calls), 1)

        # mock_get_forward_returns should be called with the initial_universe
        get_forward_returns_call = mock_get_forward_returns.mock_calls[0]

        _, args, kwargs = get_forward_returns_call

        self.assertEqual(kwargs["initial_universe"], initial_universe)

        # repeat with segmented backtest
        with patch("alphalens.pipeline.run_pipeline", mock_run_pipeline):

            from_pipeline(pipeline, "2018-01-01", "2018-01-02", factor="close",
                          quantiles=1, segment="D")

        # skip the preflight call to get_forward_returns
        self.assertEqual(len(mock_get_forward_returns.mock_calls), 3)

        get_forward_returns_call = mock_get_forward_returns.mock_calls[-1]
        _, args, kwargs = get_forward_returns_call
        self.assertEqual(kwargs["initial_universe"], initial_universe)

class ProgressMeterTestCase(TestCase):

    @skipIf(not quantrocket_installed, "quantrocket not installed")
    def test_progress_meter(self):

        date_segments = segmented_date_range("2018-01-01", "2022-11-20", segment="Y")

        self.assertListEqual(
            date_segments,
            [('2018-01-01', '2018-12-30'),
            ('2018-12-31', '2019-12-30'),
            ('2019-12-31', '2020-12-30'),
            ('2020-12-31', '2021-12-30'),
            ('2021-12-31', '2022-11-20')]
        )

        progress_meter = SegmentedAnalysisProgressMeter(date_segments)
        self.assertEqual(
            progress_meter.get_message(),
"""------------------------------  0%

[ --- ] 2018-01-01 to 2018-12-30
[ --- ] 2018-12-31 to 2019-12-30
[ --- ] 2019-12-31 to 2020-12-30
[ --- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2018-01-01', '2018-12-30'), 1)
        self.assertEqual(
            progress_meter.get_message(),
"""███---------------------------  7%

[ █-- ] 2018-01-01 to 2018-12-30
[ --- ] 2018-12-31 to 2019-12-30
[ --- ] 2019-12-31 to 2020-12-30
[ --- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2018-01-01', '2018-12-30'), 2)
        self.assertEqual(
            progress_meter.get_message(),
"""███--------------------------- 13%

[ ██- ] 2018-01-01 to 2018-12-30
[ --- ] 2018-12-31 to 2019-12-30
[ --- ] 2019-12-31 to 2020-12-30
[ --- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2018-01-01', '2018-12-30'), 3)
        self.assertEqual(
            progress_meter.get_message(),
"""██████------------------------ 20%

[  ✓  ] 2018-01-01 to 2018-12-30
[ --- ] 2018-12-31 to 2019-12-30
[ --- ] 2019-12-31 to 2020-12-30
[ --- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2018-12-31', '2019-12-30'), 1)
        self.assertEqual(
            progress_meter.get_message(),
"""█████████--------------------- 27%

[  ✓  ] 2018-01-01 to 2018-12-30
[ █-- ] 2018-12-31 to 2019-12-30
[ --- ] 2019-12-31 to 2020-12-30
[ --- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2018-12-31', '2019-12-30'), 3)
        progress_meter.mark_segment_stage(('2019-12-31', '2020-12-30'), 3)
        progress_meter.mark_segment_stage(('2020-12-31', '2021-12-30'), 2)
        self.assertEqual(
            progress_meter.get_message(),
"""█████████████████████--------- 73%

[  ✓  ] 2018-01-01 to 2018-12-30
[  ✓  ] 2018-12-31 to 2019-12-30
[  ✓  ] 2019-12-31 to 2020-12-30
[ ██- ] 2020-12-31 to 2021-12-30
[ --- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2020-12-31', '2021-12-30'), 3)
        progress_meter.mark_segment_stage(('2021-12-31', '2022-11-20'), 1)
        self.assertEqual(
            progress_meter.get_message(),
"""███████████████████████████--- 87%

[  ✓  ] 2018-01-01 to 2018-12-30
[  ✓  ] 2018-12-31 to 2019-12-30
[  ✓  ] 2019-12-31 to 2020-12-30
[  ✓  ] 2020-12-31 to 2021-12-30
[ █-- ] 2021-12-31 to 2022-11-20"""
        )

        progress_meter.mark_segment_stage(('2021-12-31', '2022-11-20'), 3)
        self.assertEqual(
            progress_meter.get_message(),
"""██████████████████████████████ 100%

[  ✓  ] 2018-01-01 to 2018-12-30
[  ✓  ] 2018-12-31 to 2019-12-30
[  ✓  ] 2019-12-31 to 2020-12-30
[  ✓  ] 2020-12-31 to 2021-12-30
[  ✓  ] 2021-12-31 to 2022-11-20"""
        )

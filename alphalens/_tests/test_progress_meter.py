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
try:
    from quantrocket.utils import segmented_date_range
    quantrocket_installed = True
except ImportError:
    quantrocket_installed = False
from ..pipeline import SegmentedAnalysisProgressMeter


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

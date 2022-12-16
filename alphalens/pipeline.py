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

import pandas as pd
from IPython.display import clear_output
try:
    from zipline.research import run_pipeline, get_forward_returns
    zipline_installed = True
except ImportError:
    zipline_installed = False
try:
    from quantrocket.utils import segmented_date_range
    quantrocket_installed = True
except ImportError:
    quantrocket_installed = False
from .tears import create_full_tear_sheet
from .utils import get_clean_factor


class EmptyPipeline(Exception):
    pass

def from_pipeline(
    pipeline,
    start_date,
    end_date,
    bundle=None,
    factor=None,
    periods=None,
    groupby=None,
    group_neutral=False,
    quantiles=None,
    bins=None,
    groupby_labels=None,
    long_short=True,
    max_loss=0.35,
    zero_aware=False,
    segment=None):
    """
    Create a full tear sheet from a zipline Pipeline. This is a shortcut for
    separately calling:

    - `zipline.research.run_pipeline`
    - `zipline.research.get_forward_returns`
    - `alphalens.utils.get_clean_factor`
    - `alphalens.tears.create_full_tear_sheet`

    Parameters
    ----------
    pipeline : zipline.pipeline.Pipeline, required
        The pipeline to run.

    start_date : str (YYYY-MM-DD), required
        First date on which the pipeline should run. If start_date is not a trading
        day, the pipeline will start on the first trading day after start_date.

    end_date : str (YYYY-MM-DD), required
        Last date on which the pipeline should run. If end_date is not a trading
        day, the pipeline will end on the first trading day after end_date. Note
        that end_date must be early enough to allow forward returns to be calculated
        according to the periods argument.

    bundle : str, optional
        the bundle code. If omitted, the default bundle will be used (and must be set).

    factor : str, required
        name of Pipeline column containing the factor to analyze.

    periods : int or list of int
        The period(s) over which to calculate forward returns.
        Example: [1, 5, 21]. Defaults to [1].

    groupby : str or list of str, optional
        name of one or more Pipeline columns to use for grouping. If provided,
        graphs will be displayed separately for each group.

    group_neutral : bool
        If True, compute quantile buckets separately for each group,
        demean returns on the group level, and weight each group
        the same in cumulative returns plots. This is useful when the
        factor values vary considerably across groups so that it
        is wise to make the binning group-relative. Cannot be used
        with multiple groupby columns. Default False.

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        10 for deciles, 4 for quartiles, etc. Alternately array of quantiles,
        e.g. [0, .25, .5, .75, 1.] for quartiles. Default is 5. Uses `pandas.qcut`
        under the hood, which chooses bin edges so that each bin has the same
        number of items. Only one of 'quantiles' or 'bins' can be specified.

    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Uses `pandas.cut` under the hood, which chooses the bin edges to be
        evenly spaced according to the values themselves, but not necessarily
        having the same number of items in each bucket. Alternately can be a
        sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]. Only one of 'quantiles' or 'bins' can be
        specified. Using 'bins' instead of 'quantiles' is useful when the factor
        contains discrete values, such a numeric score of 1-5 when want to see
        results by score.

    groupby_labels : dict or list of dict, optional
        A dictionary keyed by group code with values corresponding
        to the display name for each group. If groupby is a list or tuple,
        grouby_labels must be a list or tuple of the same length.

    long_short : bool
        Simulate a long/short portfolio if True (the default), otherwise a
        long-only portfolio. If True (long/short portfolio), relative returns
        (that is, relative to the overall mean) will be displayed, thus
        emphasizing the return spread between different quantiles. If False
        (long-only portfolio), actual returns will be displayed. Default True.

    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having forward returns for all factor values, or
        because it is not possible to perform binning. Default is 0.35.
        Set max_loss=0 to avoid Exceptions suppression.

    zero_aware : bool
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively. Default
        False.

    segment : str, optional
        run pipeline in date segments of this size, to reduce memory usage
        (use Pandas frequency string, e.g. 'Y' for yearly segments or 'Q'
        for quarterly segments). The resulting partial pipeline outputs will
        be concatenated together to produce a single tear sheet, the same
        as if this option were not used.

    Examples
    --------
    Create a tear sheet from a Pipeline that calculates 1-year momentum, grouping
    by sector and running in 1-year segments to reduce memory usage:

        from zipline.pipeline import Pipeline
        from zipline.pipeline.factors import Returns
        from zipline.pipeline.data import master
        import alphalens as al

        pipeline = Pipeline(
            columns={
                "momentum": Returns(window_length=252),
                "sector": master.SecuritiesMaster.usstock_Sector.latest
            }
        )

        al.from_pipeline(
            pipeline,
            start_date="2010-01-01",
            end_date="2022-12-31",
            periods=[1, 5, 21],
            factor="momentum",
            quantiles=5,
            groupby="sector",
            segment="Y"
        )
    """
    if not zipline_installed:
        raise ImportError("zipline must be installed to use this function")

    if segment and not quantrocket_installed:
        raise ImportError("quantrocket must be installed to use segment")

    if not start_date or not end_date:
        raise ValueError("start_date and end_date are both required")

    if factor not in pipeline.columns.keys():
        raise ValueError(
            f"invalid factor: pipeline has no column named {factor}")

    groupby_cols = groupby
    if groupby_cols:
        if not isinstance(groupby_cols, (list, tuple)):
            groupby_cols = [groupby_cols]

        for groupby_col in groupby_cols:
            if groupby_col not in pipeline.columns.keys():
                raise ValueError(
                    f"invalid groupby: pipeline has no column named {groupby_col}")

        if group_neutral and len(groupby_cols) > 1:
            raise ValueError("to use group_neutral, only one groupby column can be passed")

    else:
        groupby_cols = []

    if not periods:
        periods = [1]

    if not isinstance(periods, (list, tuple)):
        periods = [periods]

    if quantiles is None and bins is None:
        quantiles = 5

    if segment:

        factor_data = []

        date_segments = segmented_date_range(start_date, end_date, segment)

        progress_meter = SegmentedAnalysisProgressMeter(date_segments)

        for (period_start_date, period_end_date) in date_segments:

            partial_factor_data = _run_segment(
                pipeline,
                start_date=period_start_date,
                end_date=period_end_date,
                bundle=bundle,
                factor=factor,
                periods=periods,
                groupby_cols=groupby_cols,
                binning_by_group=group_neutral,
                quantiles=quantiles,
                bins=bins,
                groupby_labels=groupby_labels,
                max_loss=max_loss,
                zero_aware=zero_aware,
                progress_meter=progress_meter)

            factor_data.append(partial_factor_data)
            del partial_factor_data

        factor_data = pd.concat(factor_data, sort=True).sort_index()
        factor_data = factor_data[~factor_data.index.duplicated(keep='first')]

    else:
        factor_data = _run_segment(
            pipeline,
            start_date,
            end_date=end_date,
            bundle=bundle,
            factor=factor,
            periods=periods,
            groupby_cols=groupby_cols,
            binning_by_group=group_neutral,
            quantiles=quantiles,
            bins=bins,
            groupby_labels=groupby_labels,
            max_loss=max_loss,
            zero_aware=zero_aware)

    clear_output()
    create_full_tear_sheet(
        factor_data,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=bool(groupby_cols),
        group_name=groupby_cols or "group"
    )

def _run_segment(pipeline,
    start_date,
    end_date=None,
    bundle=None,
    factor=None,
    periods=None,
    groupby_cols=None,
    binning_by_group=False,
    quantiles=5,
    bins=None,
    groupby_labels=None,
    max_loss=0.35,
    zero_aware=False,
    progress_meter=None):

    if progress_meter:
        print(progress_meter.get_message())
        clear_output(wait=True)

    factor_data = run_pipeline(
        pipeline,
        start_date=start_date,
        end_date=end_date,
        bundle=bundle)

    if factor_data.empty:
        raise EmptyPipeline(
            "cannot create tear sheet` because the pipeline result is empty, "
            "please check the pipeline definition")

    # validate groupby columns now that we have them
    group_names = []
    groupby_series = []
    max_uniques = 50
    for groupby_col in groupby_cols:
        group_names.append(groupby_col)
        groupby_data = factor_data[groupby_col]
        num_uniques = len(groupby_data.unique())
        if num_uniques > max_uniques:
            raise ValueError(
                f"groupby column '{groupby_col}' has {num_uniques} unique values, "
                "which is too many and won't plot well. Consider using "
                "`my_factor.quantiles(5)` to group the values into a smaller number "
                f"of quantiles (maximum {max_uniques} unique values)."
            )
        groupby_series.append(groupby_data)

    if progress_meter:
        progress_meter.mark_segment_stage((start_date, end_date), 1)
        print(progress_meter.get_message())
        clear_output(wait=True)

    forward_returns = get_forward_returns(
        factor_data,
        periods=periods,
        bundle=bundle
    )

    if progress_meter:
        progress_meter.mark_segment_stage((start_date, end_date), 2)
        print(progress_meter.get_message())
        clear_output(wait=True)

    factor_data = get_clean_factor(
        factor_data[factor],
        forward_returns,
        groupby=groupby_series or None,
        binning_by_group=binning_by_group,
        quantiles=quantiles,
        bins=bins,
        groupby_labels=groupby_labels,
        group_name=group_names or "group",
        max_loss=max_loss,
        print_loss=False,
        zero_aware=zero_aware
    )

    if progress_meter:
        progress_meter.mark_segment_stage((start_date, end_date), 3)
        print(progress_meter.get_message())
        clear_output(wait=True)

    return factor_data

class SegmentedAnalysisProgressMeter:
    """
    Logging looks like this:

    ████████████------------------ 40%

    [  ✓  ] 2013-12-31 to 2014-12-30
    [  ✓  ] 2014-12-31 to 2015-12-30
    [ ██- ] 2015-12-31 to 2016-12-30
    [ --- ] 2016-12-31 to 2017-12-30
    [ --- ] 2017-12-31 to 2018-12-30
    """

    def __init__(self, date_segments):
        self.uncompleted_segments = date_segments.copy()
        self.num_segments = len(date_segments)
        self.completed_segments = []
        self.current_segment_stage = 0

    def mark_segment_stage(self, date_segment, stage):
        self.current_segment_stage = stage
        if stage > 3:
            raise ValueError("stage must be 0-3")
        elif stage == 3:
            self.uncompleted_segments.remove(date_segment)
            self.completed_segments.append(date_segment)
            self.current_segment_stage = 0

    def get_message(self):

        num_completed = len(self.completed_segments) + self.current_segment_stage / 3
        progress = num_completed / self.num_segments
        progress100 = int(round(progress, 2) * 100)
        progress30 = int(round(progress, 1) * 30)
        bar = '█' * progress30 + '-' * (30-progress30)
        progress = f'{bar} {str(progress100).rjust(2)}%'

        msg_parts = [progress]
        msg_parts.append(
            ""
        )
        for (start_date, end_date) in self.completed_segments:
            msg_parts.append(
                f"[  ✓  ] {start_date} to {end_date}")

        if self.uncompleted_segments:
            start_date, end_date = self.uncompleted_segments[0]
            bar = '█' * self.current_segment_stage + '-' * (3-self.current_segment_stage)
            msg_parts.append(
                f"[ {bar} ] {start_date} to {end_date}")

        for (start_date, end_date) in self.uncompleted_segments[1:]:
            msg_parts.append(
                f"[ --- ] {start_date} to {end_date}")

        return "\n".join(msg_parts)

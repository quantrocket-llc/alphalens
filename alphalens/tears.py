#
# Copyright 2017 Quantopian, Inc.
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
"""
Functions for creating tear sheets.

Functions
---------
create_summary_tear_sheet
    Create a small summary tear sheet with returns, information, and turnover
    analysis.

create_returns_tear_sheet
     Create a tear sheet for returns analysis of a factor.

create_information_tear_sheet
    Create a tear sheet for information analysis of a factor.

create_event_study_tear_sheet
    Create an event study tear sheet for analysis of a specific event.

create_event_returns_tear_sheet
    Create a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

create_full_tear_sheet
    Generate a number of tear sheets that are useful for analyzing a
    strategy's performance.

create_turnover_tear_sheet
    Create a tear sheet for analyzing the turnover properties of a factor.
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import plotting
from . import performance as perf
from . import utils

__all__ = [
    "create_summary_tear_sheet",
    "create_returns_tear_sheet",
    "create_information_tear_sheet",
    "create_event_study_tear_sheet",
    "create_event_returns_tear_sheet",
    "create_full_tear_sheet",
    "create_turnover_tear_sheet",
]

class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


@plotting.customize
def create_summary_tear_sheet(
    factor_data: pd.DataFrame,
    relative_returns: bool = True,
    group_neutral: bool = False,
    group_name: str = "group"
    ) -> None:
    """
    Create a small summary tear sheet with returns, information, and turnover
    analysis.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    relative_returns : bool
        If True, relative returns (that is, relative to the overall mean) will
        be displayed, thus emphasizing the return spread between different
        quantiles. If False, actual returns will be displayed. Default True.

    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.

    group_name : str, optional
        name of the group column in factor_data. Defaults to "group".
    """

    # Returns Analysis
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=relative_returns,
        group_adjust=group_neutral,
        group_name=group_name
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=relative_returns,
        group_adjust=group_neutral,
        group_name=group_name
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data,
        demeaned=relative_returns,
        group_adjust=group_neutral,
        group_name=group_name
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    periods = utils.get_forward_returns_columns(factor_data.columns)
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_factor_distribution_table(factor_data)

    plotting.plot_returns_table(
        alpha_beta,
        mean_quant_rateret,
        mean_ret_spread_quant,
        demeaned=relative_returns
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        demeaned=relative_returns,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_information_table(ic)

    # Turnover Analysis
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    plt.show()
    gf.close()


@plotting.customize
def create_returns_tear_sheet(
    factor_data: pd.DataFrame,
    factor_name: str = 'Factor',
    relative_returns: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    group_name: str = "group",
    zero_aware: bool = False,
    ) -> None:
    """
    Create a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    factor_name : str, optional
        Name of the factor. This will be used in the factor tear sheet plots and
        tables.

    relative_returns : bool
        If True, show relative returns by demeaning across the factor universe.
        If False, show actual returns. Default True.

    group_neutral : bool
        If True, demean returns on the group level, and weight each group
        the same in cumulative returns plots.

    by_group : bool
        If True, display graphs separately for each group.

    group_name : str or list of str, optional
        name of the group column in factor_data. Defaults to "group". A list
        of names can be passed to display group-level graphs based on multiple
        columns.

    zero_aware : bool
        If True, in the cumulative return plot, positive factor values will be
        longed and negative values will shorted. If False, factor values will be
        demeaned to determine long and short signals, that is, all factor values
        above the mean will be longed and all factor values below the mean will
        be shorted. Default False.
    """
    group_names = group_name
    if not isinstance(group_names, (list, tuple)):
        group_names = [group_names]
    group_name = group_names[0] if group_names else "group"

    if group_neutral and len(group_names) > 1:
        raise ValueError("to use group_neutral, only one group_name can be passed")

    factor_returns = perf.factor_returns(
        factor_data,
        demeaned=not zero_aware,
        group_adjust=group_neutral,
        group_name=group_name
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=relative_returns,
        group_adjust=group_neutral,
        group_name=group_name
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    actual_mean_quant_ret_bydate, actual_std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=False
    )

    if relative_returns:
        demeaned_mean_quant_ret_bydate, demeaned_std_quant_daily = perf.mean_return_by_quantile(
            factor_data,
            by_date=True,
            by_group=False,
            demeaned=True,
            group_adjust=group_neutral,
            group_name=group_name
        )

    mean_quant_ret_bydate, std_quant_daily = (
        (demeaned_mean_quant_ret_bydate, demeaned_std_quant_daily)
        if relative_returns else
        (actual_mean_quant_ret_bydate, actual_std_quant_daily)
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data,
        factor_returns
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    fr_cols = len(factor_returns.columns)
    fr_1d_cols = [col for col in factor_returns.columns
                  if col.lower() in ("1d", "overnight", "intraday")]
    vertical_sections_per_1d_col = 3 if relative_returns else 2
    vertical_sections = (
        # bar plot and violin plot
        2
        # cumulative returns plots
        + (len(fr_1d_cols) * vertical_sections_per_1d_col)
        # top - bottom mean quantile returns
        + fr_cols
        )
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_returns_table(
        alpha_beta,
        mean_quant_rateret,
        mean_ret_spread_quant,
        demeaned=relative_returns
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        demeaned=relative_returns,
        ylim_percentiles=None,
        ax=gf.next_row(),
        factor_name=factor_name,
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate,
        demeaned=relative_returns,
        ylim_percentiles=(1, 99),
        ax=gf.next_row(),
        factor_name=factor_name,
    )

    # Compute cumulative returns from daily simple returns, if '1D', 'Intraday',
    # or 'Overnight' returns are provided.
    for colname in fr_1d_cols:

        title = (
            f"{factor_name}-Weighted"
            + (f", {group_name}-Neutral," if group_neutral else "")
            + f" %s Portfolio Cumulative Return ({colname} Period)"
        )

        plotting.plot_cumulative_returns(
            factor_returns[[colname]].rename(columns={colname: "Long/Short"}),
            period=colname,
            color=sns.color_palette('colorblind')[3],
            title=title % "Long/Short",
            ax=gf.next_row()
        )

        if relative_returns:
            plotting.plot_cumulative_returns_by_quantile(
                demeaned_mean_quant_ret_bydate[colname],
                period=colname,
                relative_or_actual="Relative",
                group_neutral_name=group_name if group_neutral else None,
                ax=gf.next_row(),
                factor_name=factor_name,
            )

        plotting.plot_cumulative_returns_by_quantile(
            actual_mean_quant_ret_bydate[colname],
            period=colname,
            # say "Actual" Cumulative Return if there's a "Relative"
            # plot, but otherwise nothing
            relative_or_actual="Actual" if relative_returns else "",
            ax=gf.next_row(),
            factor_name=factor_name
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()

    if by_group:
        for group_name in group_names:

            if group_name in factor_data.select_dtypes("category").columns:
                factor_data[group_name] = factor_data[group_name].cat.rename_categories({
                    '': '<blank>'})
            (
                mean_return_quantile_group,
                mean_return_quantile_group_std_err,
            ) = perf.mean_return_by_quantile(
                factor_data,
                by_date=False,
                by_group=True,
                group_name=group_name,
                demeaned=relative_returns,
                group_adjust=group_neutral,
            )

            mean_quant_rateret_group = mean_return_quantile_group.apply(
                utils.rate_of_return,
                axis=0,
                base_period=mean_return_quantile_group.columns[0],
            )

            num_groups = len(
                mean_quant_rateret_group.index.get_level_values(group_name).unique()
            )

            vertical_sections_per_mini_plot_type = ((num_groups - 1) // 2) + 1

            num_mini_plots = (
                # mean return bar plots are always shown
                1
                # for each 1D column, we show an actual cumulative return plot
                + len(fr_1d_cols)
                # and a relative cumulative return plot, if relative_returns=True
                + (len(fr_1d_cols) if relative_returns else 0)
            )

            vertical_sections = (
                # bottom and top quantile composition by group
                1
                + vertical_sections_per_mini_plot_type * num_mini_plots
            )
            gf = GridFigure(rows=vertical_sections, cols=2)

            plotting.plot_quantile_composition_by_group(
                factor_data,
                group_name=group_name,
                ax=[gf.next_cell(), gf.next_cell()],
                factor_name=factor_name)

            ax_quantile_returns_bar_by_group = [
                gf.next_cell() for _ in range(num_groups)
            ]
            plotting.plot_quantile_returns_bar(
                mean_quant_rateret_group,
                by_group=True,
                group_name=group_name,
                demeaned=relative_returns,
                ylim_percentiles=(5, 95),
                ax=ax_quantile_returns_bar_by_group,
                factor_name=factor_name
            )

            for colname in fr_1d_cols:

                if relative_returns:
                    (
                        demeaned_mean_quant_group_ret_by_date,
                        _,
                    ) = perf.mean_return_by_quantile(
                        factor_data,
                        by_date=True,
                        by_group=True,
                        group_name=group_name,
                        demeaned=True,
                        group_adjust=group_neutral,
                    )

                    ax_quantile_cumulative_returns_by_group = [
                        gf.next_cell() for _ in range(num_groups)
                    ]

                    plotting.plot_cumulative_returns_by_quantile(
                        demeaned_mean_quant_group_ret_by_date[colname],
                        period=colname,
                        by_group=True,
                        group_name=group_name,
                        relative_or_actual="Relative",
                        group_neutral_name=None,
                        ax=ax_quantile_cumulative_returns_by_group,
                        factor_name=factor_name
                    )

                (
                    actual_mean_quant_group_ret_by_date,
                    _,
                ) = perf.mean_return_by_quantile(
                    factor_data,
                    by_date=True,
                    by_group=True,
                    group_name=group_name,
                    demeaned=False,
                    group_adjust=group_neutral,
                )

                ax_quantile_cumulative_returns_by_group = [
                    gf.next_cell() for _ in range(num_groups)
                ]

                plotting.plot_cumulative_returns_by_quantile(
                    actual_mean_quant_group_ret_by_date[colname],
                    period=colname,
                    by_group=True,
                    group_name=group_name,
                    # say "Actual" Cumulative Return if there's a "Relative"
                    # plot, but otherwise nothing
                    relative_or_actual="Actual" if relative_returns else "",
                    ax=ax_quantile_cumulative_returns_by_group,
                    factor_name=factor_name
                )
            plt.show()
            gf.close()


@plotting.customize
def create_information_tear_sheet(
    factor_data: pd.DataFrame,
    group_neutral: bool = False,
    by_group: bool = False,
    group_name: str = "group"
    ) -> None:
    """
    Create a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    group_neutral : bool
        Demean forward returns by group before computing IC.

    by_group : bool
        If True, display graphs separately for each group.

    group_name : str or list of str, optional
        name of the group column in factor_data. Defaults to "group". A list
        of names can be passed to display group-level graphs based on multiple
        columns.
    """
    if not isinstance(group_name, (list, tuple)):
        group_names = [group_name]
    else:
        group_names = group_name
        group_name = group_names[0] if group_names else "group"

    if group_neutral and len(group_names) > 1:
        raise ValueError("to use group_neutral, only one group_name can be passed")

    ic = perf.factor_information_coefficient(
        factor_data,
        group_adjust=group_neutral,
        group_name=group_name)

    plotting.plot_information_table(ic)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    vertical_sections += len(group_names) - 1
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:

        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            group_name=group_name,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            mean_monthly_ic, ax=ax_monthly_ic_heatmap
        )

    if by_group:
        for group_name in group_names:
            mean_group_ic = perf.mean_information_coefficient(
                factor_data,
                group_adjust=group_neutral,
                by_group=True,
                group_name=group_name
            )

            plotting.plot_ic_by_group(mean_group_ic, group_name=group_name, ax=gf.next_row())

    plt.show()
    gf.close()


@plotting.customize
def create_turnover_tear_sheet(
    factor_data: pd.DataFrame,
    turnover_periods: list[str] = None
    ) -> None:
    """
    Create a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).values
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    if not turnover_periods:
        return

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()


@plotting.customize
def create_full_tear_sheet(
    factor_data: pd.DataFrame,
    factor_name: str = 'Factor',
    relative_returns: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    group_name: str = "group",
    zero_aware: bool = False,
    ) -> None:
    """
    Create a full tear sheet for analysis and evaluation of a single
    return-predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    factor_name : str, optional
        Name of the factor. This will be used in the factor tear sheet plots and
        tables.

    relative_returns : bool
        If True, relative returns (that is, relative to the overall mean) will
        be displayed, thus emphasizing the return spread between different
        quantiles. If False, actual returns will be displayed. Default True.

    group_neutral : bool
        If True, demean returns on the group level, and weight each group
        the same in cumulative returns plots.

    by_group : bool
        If True, display graphs separately for each group.

    group_name : str or list of str, optional
        name of the group column in factor_data. Defaults to "group". A list
        of names can be passed to display group-level graphs based on multiple
        columns.

    zero_aware : bool
        If True, in the cumulative return plot, positive factor values will be
        longed and negative values will shorted. If False, factor values will be
        demeaned to determine long and short signals, that is, all factor values
        above the mean will be longed and all factor values below the mean will
        be shorted. Default False.
    """

    plotting.plot_factor_distribution_table(factor_data, factor_name=factor_name)
    create_returns_tear_sheet(
        factor_data,
        factor_name,
        relative_returns,
        group_neutral,
        by_group,
        group_name=group_name,
        zero_aware=zero_aware,
        set_context=False
    )
    create_information_tear_sheet(
        factor_data, group_neutral, by_group, group_name=group_name, set_context=False
    )
    create_turnover_tear_sheet(factor_data, set_context=False)


@plotting.customize
def create_event_returns_tear_sheet(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    avgretplot: tuple[int, int] = (5, 15),
    relative_returns: bool = True,
    group_neutral: bool = False,
    std_bar: bool = True,
    by_group: bool = False,
    group_name: str = "group"
    ) -> None:
    """
    Create a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns

    relative_returns : bool
        If True, relative returns (that is, relative to the overall mean) will
        be displayed, thus emphasizing the return spread between different
        quantiles. If False, actual returns will be displayed. Default True.

    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.

    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile

    by_group : bool
        If True, display graphs separately for each group.

    group_name : str, optional
        name of the group column in factor_data. Defaults to "group".
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=relative_returns,
        group_adjust=group_neutral,
        group_name=group_name
    )

    num_quantiles = int(factor_data["factor_quantile"].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += ((num_quantiles - 1) // 2) + 1
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
        ax=gf.next_row(),
    )
    if std_bar:
        ax_avg_cumulative_returns_by_q = [
            gf.next_cell() for _ in range(num_quantiles)
        ]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q,
        )

    plt.show()
    gf.close()

    if by_group:
        groups = factor_data["group"].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=relative_returns,
            group_adjust=group_neutral,
            by_group=True,
            group_name=group_name
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level=group_name):
            avg_cumret.index = avg_cumret.index.droplevel(group_name)
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell(),
            )

        plt.show()
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    avgretplot: tuple[int, int] = (5, 15),
    rate_of_ret: bool = True,
    n_bars: int = 50
    ) -> None:
    """
    Create an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.

        - See full explanation in :class:`alphalens.utils.get_clean_factor_and_forward_returns`

    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).

    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots

    n_bars : int, optional
        Number of bars in event distribution plot
    """

    relative_returns = False

    plotting.plot_factor_distribution_table(factor_data)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row()
    )
    plt.show()
    gf.close()

    if returns is not None and avgretplot is not None:

        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            relative_returns=relative_returns,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(
        factor_data,
        demeaned=False,
        equal_weight=True
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=relative_returns
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=relative_returns
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections + 1, cols=1)

    plotting.plot_quantile_returns_bar(
        mean_quant_ret,
        by_group=False,
        demeaned=relative_returns,
        ylim_percentiles=None,
        ax=gf.next_row()
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate,
        demeaned=relative_returns,
        ylim_percentiles=(1, 99),
        ax=gf.next_row()
    )

    plt.show()
    gf.close()

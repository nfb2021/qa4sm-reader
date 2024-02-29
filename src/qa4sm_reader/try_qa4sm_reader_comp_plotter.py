# %%
import sys

sys.path.append(
    '/home/nbader/Documents/QA4SM_tasks/jira-744/qa4sm-reader/src/')

from qa4sm_reader.plotter import QA4SMCompPlotter
from qa4sm_reader.plotting_methods import ClusteredBoxPlot, patch_styling
from qa4sm_reader.handlers import QA4SMVariable, MixinVarmeta, MetricVariable
import qa4sm_reader.globals as globals
import os
from icecream import ic
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from matplotlib.patches import Rectangle
from typing import List, Dict

METRIC = 'R'

pth = os.path.join(
    '/home/nbader/Documents/QA4SM_tasks/jira-744/qa4sm/output/07977749-a7f7-4a8b-b870-b2e6d88870e0',
    '0-ISMN.soil_moisture_with_1-SMOS_L3.Soil_Moisture_with_2-ESA_CCI_SM_active.sm_with_3-ERA5.swvl1.nc'
)

p = QA4SMCompPlotter(pth)


def get_metric_vars(generic_metric: str) -> Dict[str, str]:
    _dict = {}

    def get_dc_attrs(dc_name: str) -> Dict[str, str]:
        _list = [
            attr for attr in p.ds.attrs
            if attr.endswith(dc_name.split('-')[0])
        ]
        return {attr: p.ds.attrs[attr][:-1] for attr in _list}

    for dataset in p.get_datasets_from_df(df):
        for ds_combi, specific_metric in p.metrics_ds_grouped_lut(
        )[generic_metric].items():
            if dataset in ds_combi:
                _Var = MetricVariable(varname=specific_metric,
                                      global_attrs=p.ds.attrs)
                if _Var.values == None:
                    _Var.values = df.loc[:, (dataset, slice(None))]

                _dict[dataset] = _Var

    return _dict


def get_legend_entries(generic_metric: str) -> Dict[str, str]:
    return {
        f'{Var.metric_ds[0]}-{Var.metric_ds[1]["short_name"]}':
        # 'hello':
        cbp.label_template.format(
            dataset_name=Var.metric_ds[1]["pretty_name"],
            dataset_version=Var.metric_ds[1]
            ["pretty_version"],  # Replace with your actual dataset version
            variable_name=Var.metric_ds[1]
            ["pretty_variable"],  # Replace with your actual variable name
            unit=Var.metric_ds[1]["mu"])
        for Var in get_metric_vars(generic_metric).values()
    }


#%%

# dfs = [
#     df.loc[:, (dataset, slice(None))] for dataset in p.get_datasets_from_df(df)
# ]
df = p.get_metric_df(METRIC)
Vars = get_metric_vars(METRIC)
anchor_list = np.linspace(1, len(p.tsws_used), len(p.tsws_used))
cbp = ClusteredBoxPlot(
    anchor_list=anchor_list,
    no_of_ds=3,
)
cbp
legend_entries = get_legend_entries(METRIC)

# %%
centers_and_widths = cbp.centers_and_widths(anchor_list=cbp.anchor_list,
                                            no_of_ds=cbp.no_of_ds,
                                            space_per_box_cluster=0.9,
                                            rel_indiv_box_width=0.8)
centers_and_widths
# %%
cbp_fig = cbp.figure_template()
colors = ['blue', 'red', 'gold']

#%%
legend_handles = []
for dc_num, (dc_val_name, Var) in enumerate(Vars.items()):
    df = Var.values
    ic(df)
    bp = cbp_fig.ax_box.boxplot(
        df.dropna().values,
        positions=centers_and_widths[dc_num].centers,
        widths=centers_and_widths[dc_num].widths,
        showfliers=False,
        patch_artist=True,
    )

    for box in bp['boxes']:
        box.set(color=colors[dc_num])

    legend_handles.append(
        Rectangle((0, 0),
                  1,
                  1,
                  color=colors[dc_num],
                  label=legend_entries[dc_val_name]))

    patch_styling(bp, colors[dc_num])

    # cbp_fig.ax_median.plot(
    #     centers_and_widths[ds].centers,
    #     df.median(),
    #     marker='o',
    #     linestyle='None',
    #     color=colors[ds],
    # )
    # cbp_fig.ax_iqr.plot(
    #     centers_and_widths[ds].centers,
    #     df.quantile(0.75) - df.quantile(0.25),
    #     marker='o',
    #     linestyle='None',
    #     color=colors[ds],
    # )
    # cbp_fig.ax_n.plot(
    #     centers_and_widths[ds].centers,
    #     df.count(),
    #     marker='o',
    #     linestyle='None',
    #     color=colors[ds],
    # )

cbp_fig.ax_box.legend(handles=legend_handles,
                      loc='upper right',
                      ncols=cbp.no_of_ds)
cbp_fig.ax_box.set_ylabel(METRIC)
# cbp_fig.ax_box.set_xlabel('Temporal sub-windows')
cbp_fig.ax_box.set_title('Clustered Boxplot')

# cbp_fig.ax_median.set_ylabel('Median')
# cbp_fig.ax_iqr.set_ylabel('IQR')
# cbp_fig.ax_n.set_ylabel('N')
# cbp_fig.ax_n.set_xlabel('Temporal sub-windows')

xtick_pos = cbp.centers_and_widths(anchor_list=cbp.anchor_list,
                                   no_of_ds=1,
                                   space_per_box_cluster=0.7,
                                   rel_indiv_box_width=0.8)
cbp_fig.ax_box.set_xticks([])
cbp_fig.ax_box.set_xticklabels([])
cbp_fig.ax_box.set_xticks(xtick_pos[0].centers)


def get_xtick_labels(df: pd.DataFrame) -> List:
    _count_dict = df.count().to_dict()
    return [f"{tsw[1]}\nN: {count}" for tsw, count in _count_dict.items()]


ic(df.count().to_dict(), type(df.count()))

cbp_fig.ax_box.set_xticklabels(get_xtick_labels(df))
label = METRIC
if label is not None:
    plt.ylabel(label, weight='normal')
# cbp_fig.ax_box.tick_params(labelsize=globals.tick_size)
# cbp_fig.ax_box.spines['right'].set_visible(False)
# cbp_fig.ax_box.spines['top'].set_visible(False)

dummy_xticks = [
    cbp_fig.ax_box.axvline(x=(a + b) / 2, color='lightgrey')
    for a, b in zip(xtick_pos[0].centers[:-1], xtick_pos[0].centers[1:])
]

cbp_fig.fig.savefig(f'test_{METRIC}.png')

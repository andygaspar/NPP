import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter, A2
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


df = pd.read_csv('Results/arc_results.csv')
# df_gap = pd.read_csv('Results/benchmark_results.csv')
# df["mip_GAP"] = df_gap.mip_GAP
# df["obj_exact"] = df_gap.obj_exact

df.graphs = df.graphs.apply(lambda x: x[0])
df['case'] = ' ' + df.graphs + ' ' + df.n_com.astype(str) + ' ' + df.toll_pr.astype(str)
df['gap'] = (1 - df.g_obj/df.exact_obj) * 100


# df['GAP'] = 1 - df.obj_h / df.obj_exact
# df['GAP'] = df['GAP'].round(9)
# df['time_exact'] = df_gap["time_exact"]
df.columns

df_res = df.groupby('case', as_index=False).agg(
    {'g_time': ['mean', 'std'], 'exact_time': ['mean', 'std'], 'gap': ['mean', 'std'], 'MIP_gap': ['mean', 'std']})
df_res.columns
df_res.columns = ['CASE', 'GAH t mean', 'GAH t std', 'GA t mean', 'GA t std', 'Exact t mean', 'Exact t std', 'GAH/Exact GAP mean',
                  'GAH/Exact GAP t std', 'GA/Exact GAP mean', 'GA/Exact GAP t std', 'MIP gap mean', 'MIP gap  std']
df_res.columns
# df_res_tab = df_res[
#     ['CASE', 'GA t mean', 'GA t std', 'Exact t mean', 'Exact t std', 'GA/Exact GAP mean', 'GA/Exact GAP t std',
#      'MIP gap mean', 'MIP gap  std']].copy()
# df_res_tab.status = df_res_tab.status.apply(lambda s: 'opt' if s<3 else 'time_limit')
# df_res_tab = df_res_tab.astype({'commodities': int, 'paths': int})





df_tex = df_res.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

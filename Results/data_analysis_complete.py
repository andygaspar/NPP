import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter, A2
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from Instance.instance import Instance
from Solver.solver import GlobalSolver

df = pd.read_csv('Results/results.csv')
df_gap = pd.read_csv('Results/benchmark_results.csv')
df["mip_GAP"] = df_gap.mip_GAP
df["obj_exact"] = df_gap.obj_exact

df = df.astype({'run': int, 'commodities': int, 'paths': int})
df['case'] = df.commodities.astype(str) + '_' + df.paths.astype(str)
df['GAP'] = 1 - df.obj_gah / df.obj_exact
df['time_exact'] = df_gap["time_exact"]
df.columns

df_res = df.groupby('case', as_index=False).agg(
    {'time_gah': ['mean', 'std'], 'time_exact': ['mean', 'std'], 'GAP': ['mean', 'std'],
     'mip_GAP': ['mean', 'std']})
df_res.columns = ['CASE', 'GA t mean', 'GA t std', 'Exact t mean', 'Exact t std', 'GA/Exact GAP mean',
                  'GA/Exact GAP t std',
                  'MIP gap mean', 'MIP gap  std']
df_res.columns
df_res_tab = df_res[
    ['CASE', 'GA t mean', 'GA t std', 'Exact t mean', 'Exact t std', 'GA/Exact GAP mean', 'GA/Exact GAP t std',
     'MIP gap mean', 'MIP gap  std']].copy()
# df_res_tab.status = df_res_tab.status.apply(lambda s: 'opt' if s<3 else 'time_limit')
# df_res_tab = df_res_tab.astype({'commodities': int, 'paths': int})

df_res_tab["CASE"] = df_res["CASE"].apply(lambda x: 'C' + x[:2] + " P" + x[3:])
df_res_tab.options.display.precision = 3
df_res_tab.style.format = ".2f"
table_data = [list(df_res_tab.columns)]
for i, row in df_res_tab.iterrows():
    table_data.append(['{0:.4f}'.format(el) if type(el) == float else el for el in list(row)])

table = Table(table_data)
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
    ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
])

table.setStyle(table_style)
pdf_table = []
pdf_table.append(table)
pdf = SimpleDocTemplate("Results/dataframe.pdf", pagesize=A2)
pdf.build(pdf_table)

df = pd.read_csv('benchmark_results.csv')
df_1 = pd.read_csv('results.csv')

df_1['obj_exact'] = df.obj_exact
df_1['time_exact'] = df.time_exact
df_1['mip_GAP'] = df.mip_GAP * 100
df_1['gah_GAP'] = (1 - df_1.obj_gah / df_1.obj_exact) * 100
df_1['ga_GAP'] = (1 - df_1.obj_ga / df_1.obj_exact) * 100

df_1 = df_1.astype({'run': int, 'commodities': int, 'paths': int})
df_1['case'] = df_1.commodities.astype(str) + '_' + df_1.paths.astype(str)

df_res = df_1.groupby('case').mean()
df_res_tab = df_res[
    ['commodities', 'paths', 'time_exact', 'time_gah', 'time_ga', 'gah_GAP', 'ga_GAP', 'mip_GAP']].copy()
# df_res_tab.status = df_res_tab.status.apply(lambda s: 'time_exact' if s<3 else 'time_limit')
df_res_tab = df_res_tab.astype({'commodities': int, 'paths': int})
df_tex = df_res_tab.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

df_res.columns


df_s = df_res[df_res['CASE'].isin(['20_20', '56_56', '90_90'])]
plt.style.use('ggplot') #Change/Remove This If you Want

fig, ax = plt.subplots(figsize=(15, 5))
plt.rcParams["font.size"] = 15
x = [1, 2, 3]
ax.plot(x, df_s['Exact t mean'], alpha=0.5, color='red', label="Exact",linewidth = 4.0)
ax.fill_between(x, df_s['Exact t mean'] - df_s['Exact t std'], df_s['Exact t mean'] + df_s['Exact t std'],  color='blue', alpha=0.4)
ax.plot(x, df_s['GA t mean'], alpha=0.5, color='black', label="GA",linewidth = 4.0)
ax.fill_between(x, df_s['GA t mean'] - df_s['GA t std'], df_s['GA t mean'] + df_s['GA t std'], color='green', alpha=0.4)
ax.legend(loc='best')
# ax.legend(loc='best')

# ax.set_ylim([0.88,1.02])
ax.set_xticks(x, ['c20 p20', 'c56 p56', 'c90 p90'])
ax.set_ylabel("Time")
plt.tight_layout()
plt.show()


opt = df[df.GAP<0.00001]

ff = df[df['case']=='20_20']

n_paths = 20
n_commodities = 20

npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
solver = GlobalSolver(npp, verbose=True, time_limit=3)
solver.solve()
print(npp.compute_solution_value(solver.solution_array), solver.obj, 'sol')

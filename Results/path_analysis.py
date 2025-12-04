import pandas as pd

from Results.PathResults.old.path_analysis import compare_results_no_ga


def format_df(dff, df_with_MIP=None):
    dff = dff.astype({'run': int, 'commodities': int, 'paths': int})
    dff['partial'] = dff.partial.apply(lambda x: 'P' if x else 'C')
    dff['case'] = ' ' + dff.partial.astype(str) + ' ' + dff.commodities.astype(str) + ' ' + dff.paths.astype(str)
    if df_with_MIP is not None:
        dff.obj_exact = df_with_MIP.obj_exact
        dff.mip_GAP = df_with_MIP.mip_GAP * 100
        dff.time_exact = df_with_MIP.time_exact
        if 'GAP_ga' in dff.columns:
            dff['GAP_h'] = (1 - dff.obj_h / dff.obj_exact) * 100
            dff['GAP_ga'] = (1 - dff.obj_ga / dff.obj_exact) * 100
        dff.status = df_with_MIP.status
    else:
        dff.GAP_h = dff.GAP_h * 100
        if 'GAP_ga' in dff.columns:
            dff.GAP_ga = dff.GAP_ga * 100
        dff.mip_GAP = dff.mip_GAP * 100
    dff.status = dff.status.apply(lambda x: 1 if x == 2 else 0)
    dff['gah_improvement'] = (dff.GAP_h <= - 1e-9).astype(int)
    dff['gah_opt'] = (dff.GAP_h <= 1e-9).astype(int) * (dff.GAP_h >= - 1e-9).astype(int)
    if 'GAP_ga' in dff.columns:
        dff['ga_improvement'] = (dff.GAP_ga <= - 1e-9).astype(int)
        dff['ga_opt'] = (dff.GAP_ga <= 1e-9).astype(int) * (dff.GAP_ga >= - 1e-9).astype(int)
    dff['h_iter'] = dff['h_iter']/1000
    return dff


def compare_results(dff):
    dff.case_num = dff.case_num.apply(lambda x: x + 22 if x < 11 else x)
    dff = dff.groupby('case', as_index=False).agg(
        {'time_h': ['mean', 'std'], 'time_ga': ['mean', 'std'], 'time_exact': ['mean', 'std'], 'GAP_h': ['mean', 'std'],
         'GAP_ga': ['mean', 'std'],
         'mip_GAP': ['mean', 'std'], 'status': ['sum'], 'gah_opt': ['sum'], 'gah_improvement': ['sum'],
         'ga_opt': ['sum'], 'ga_improvement': ['sum'], 'h_iter': ['mean', 'std'], 'case_num': ['max']})

    dff.sort_values(by=('case_num', 'max'), ascending=True, inplace=True)
    dff.columns = ['CASE', 'GAH t mean', 'GAH t std', 'GA t mean', 'GA t std', 'Exact t mean', 'Exact t std', 'GAH/Exact GAP mean',
                   'GAH/Exact GAP std', 'GA/Exact GAP mean', 'GA/Exact GAP std', 'MIP gap mean', 'MIP gap  std', 'mip opt', 'gah opt',
                   'gah improvement', 'ga opt', 'ga improvement', 'h_iter mean', 'h_iter std', 'case_num']
    return dff


df = pd.read_csv('Results/path_results.csv')
df_0 = format_df(df)
df_res = compare_results(df_0)

print(df_0[(df_0.status == 1) & (df_0.gah_opt == 1)].shape[0])
df_tex = df_res[df_res.columns[:-5]].style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

df_tex = df_res[df_res.columns[-5:]].style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

print((df_0.obj_h - df_0.obj_ga <= - 1e-9).sum())
print(((df_0.obj_h - df_0.obj_ga >= - 1e-9) * (df_0.obj_h - df_0.obj_ga <= 1e-9)).sum())


df_exact = df[df.commodities < 180]
df_1 = pd.read_csv('Results/path_results_parameters.csv')
df_1 = format_df(df_1, df_exact)
df_11 = compare_results(df_1)
df_11.columns = [s + '1' for s in df_11.columns]

df_2 = pd.read_csv('Results/PathResults/path_results_256_10000_h_iterations.csv')
df_2 = format_df(df_2, df_exact)
df_22 = compare_results(df_2)
df_22.columns = [s + '2' for s in df_22.columns]

df_3 = pd.read_csv('Results/PathResults/path_results_256_20000_h_iterations.csv')
df_3 = format_df(df_3, df_exact)
df_33 = compare_results(df_3)
df_33.columns = [s + '3' for s in df_33.columns]

df_compar = pd.concat([df_res, df_11, df_22, df_33], axis=1)
df_compar.columns

df_comparison = df_compar[['CASE', 'GAH/Exact GAP mean', 'GAH/Exact GAP std', 'gah opt', 'gah improvement',
                           'GAH/Exact GAP mean1', 'GAH/Exact GAP std1', 'gah opt1', 'gah improvement1',
                           'GAH/Exact GAP mean2', 'GAH/Exact GAP std2', 'gah opt2', 'gah improvement2',
                           'GAH/Exact GAP mean3', 'GAH/Exact GAP std3', 'gah opt3', 'gah improvement3']]

df_comparison = df_comparison[df_comparison.columns[[0, 1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16]]]

df_tex = df_comparison.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

df_time_comparison = df_compar[['CASE', 'GA t mean', 'GA t std', 'GA t mean1', 'GA t std1', 'GA t mean2', 'GA t std2',
                                'GA t mean3', 'GA t std3', 'h_iter mean', 'h_iter std', 'h_iter mean1', 'h_iter std1', 'h_iter mean2', 'h_iter std2',
                           'h_iter mean3', 'h_iter std3']].copy()

for item in ['h_iter mean', 'h_iter std', 'h_iter mean1', 'h_iter std1', 'h_iter mean2', 'h_iter std2',
                           'h_iter mean3', 'h_iter std3']:
    df_time_comparison[item] = df_time_comparison[item].astype(int)

df_tex = df_time_comparison.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())

print(df_0.GAP_h.mean(), df_0.GAP_h.std())
print(df_1.GAP_h.mean(), df_1.GAP_h.std())
print(df_2.GAP_h.mean(), df_2.GAP_h.std())
print(df_3.GAP_h.mean(), df_3.GAP_h.std())

df_comparison.columns
df_PS = df_comparison[df_comparison.columns[:9]]
df_PS = df_PS[df_PS.CASE.isin([' C 20 90', ' P 20 90'])]
df_tex = df_PS.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())


df_PS_time = df_time_comparison[df_time_comparison.columns[:9]]
df_PS_time = df_PS_time[df_PS_time.CASE.isin([' C 20 90', ' P 20 90'])]
df_tex = df_PS_time.style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())


df_large = pd.read_csv('Results/PathResults/path_results_large.csv')
df_l = format_df(df_large)
df_res_large = compare_results_no_ga(df_l)
df_tex = df_res_large[df_res_large.columns[:-2]].style.format(decimal=',', thousands='.', precision=3).hide(axis="index")
print(df_tex.to_latex())
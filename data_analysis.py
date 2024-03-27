import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

df = pd.read_csv('test.csv')

df = df.astype({'run': int, 'commodities': int, 'paths': int})
df['case'] = df.commodities.astype(str) + '_' + df.paths.astype(str)


df_res = df.groupby('case').mean()
df_res_tab = df_res[['commodities', 'paths', 'gap', 'time_exact', 'time_alg', 'status', 'mip_gap']].copy()
df_res_tab.status = df_res_tab.status.apply(lambda s: 'opt' if s<3 else 'time_limit')
df_res_tab = df_res_tab.astype({'commodities': int, 'paths': int})

table_data = [list(df_res_tab.columns)]
for i, row in df_res_tab.iterrows():
    table_data.append(list(row))

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
pdf_table=[]
pdf_table.append(table)
pdf = SimpleDocTemplate("dataframe.pdf", pagesize=letter)
pdf.build(pdf_table)

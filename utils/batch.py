import glob
import random
import psicalc as pc
import xlsxwriter
import pandas as pd
import numpy as np
import re

dir_name = 'PSICalc Pfam Datasets for analaysis'
results_dir = 'PSICalc Pfam Results'
spread = 1
types = ['*.csv', '*.CSV', '*.txt']
f = []

for e in types:
    f.extend(glob.glob(f'{dir_name}/{e}'))
random.shuffle(f)

types = ['*.xlsx']
j = []
for e in types:
    j.extend(glob.glob(f'{results_dir}/{e}'))

remove_list = []
for i in range(len(f)):
    for l in range(len(j)):
        val1 = j[l].split('/')[1].split('.')[0]
        val2 = f[i].split('/')[1].split('.')[0]
        if val1 == val2:
            remove_list.append(f[i])

f = [x for x in f if x not in remove_list]

# Step One: Report all data for each MSA
for i in range(len(f)):
    print("\nReading: ", f[i])

    if f[i].endswith(".txt"):
        df = pc.read_txt_file_format(f[i])
    else:
        df = pc.read_csv_file_format(f[i])

    name = results_dir + '/' + f[i].split('/')[1].split('.')[0] + '.xlsx'
    workbook = xlsxwriter.Workbook(name)

    df = pc.rectify_sequences(df)
    row_ix, df = pc.deweese_schema(df)
    df = pc.remove_high_insertion_sites(df, 20)

    worksheet = workbook.add_worksheet(str(spread))
    results = pc.find_clusters(spread, df)

    row = 0
    column = 0

    worksheet.write(row, column, "Cluster")
    worksheet.write(row, column+1, "Sr_mode")
    worksheet.write(row, column+2, "Discovered")
    worksheet.write(row, column+3, row_ix)

    for key, value in results.items():
        row += 1
        val1, val2 = value
        worksheet.write(row, column, str(key))
        worksheet.write(row, column+1, str(val1))
        worksheet.write(row, column+2, str(val2))

    workbook.close()

# Step Two: Aggregate best results for n orders
results_dir = 'PSICalc Pfam Results'
row = 0
column = 0

types = ['*.xlsx']
f = []
for e in types:
    f.extend(glob.glob(f'{results_dir}/{e}'))
workbook = xlsxwriter.Workbook('results_table.xlsx')
worksheet = workbook.add_worksheet("Top results")

for i in range(len(f)):
    name = f[i].split('/')[1].split('.')[0]
    file = pd.ExcelFile(f[i])
    df1 = pd.read_excel(file, '1')
    df = df1
    sample = str(df.columns[3])
    worksheet.write(row, column, name)
    worksheet.write(row, 1, sample)
    row += 1

    x = list(df["Cluster"].values)
    a = [re.sub(r'[()]', '', j) for j in x]
    df = df.replace(df["Cluster"].values, a)

    order = 2
    while order < 6:
        maximum = 0
        cluster, sr_mode = None, None
        for index, line in df.iterrows():
            measure = str(line["Cluster"]).split(",")
            ints = [int(item) for item in measure]
            if len(ints) == order:
                if line["Sr_mode"] > maximum:
                    cluster, sr_mode = line["Cluster"], line["Sr_mode"]
        worksheet.write(row, column, "Order {}".format(order))
        worksheet.write(row, column+1, cluster)
        worksheet.write(row, column+2, sr_mode)
        order += 1
        row += 1
    row += 2

workbook.close()

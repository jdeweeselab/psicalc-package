"""
Author: Thomas Townsley
Email: thomas@mandosoft.dev

This is a script for calculating the entropy of each column
by utilizing the entropy() function in the psicalc nmi libary.
At the time of writing this is used to investigate
whether to incorporate an entropy cutoff value to overcome
the problems associated with low entropy columns normalizing
to 1 and thus producing a false high entropy cluster.
"""

import psicalc as pc
import csv
from psicalc.nmi import entropy

df = pc.read_csv_file_format('../datasets/TOP2A_protein_105species REV trunc to HTIIa d.csv')
matrix = pc.encode_msa(df)
num_rows, num_cols = matrix.shape
columns = [matrix[:, col_idx] for col_idx in range(num_cols)]
data = [['Column', 'Entropy']]
for col_idx, column in enumerate(columns):
    e = entropy(column)
    data.append([(col_idx + 1), round(e, 6)])

with open('entropy_results.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)


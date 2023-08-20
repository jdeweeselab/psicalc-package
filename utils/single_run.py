"""
Author: Thomas Townsley
Email: thomas@mandosoft.dev

This is useful for testing single runs of the program. Very
helpful when debugging.
"""

import psicalc as pc
import csv
from psicalc.nmi import entropy

df = pc.read_csv_file_format('../datasets/TOP2A_protein_105species REV trunc to HTIIa d.csv')
data = pc.durston_schema(df, 2)
result = pc.find_clusters(17, data, "pairwise_only", 0.3)
print(result)

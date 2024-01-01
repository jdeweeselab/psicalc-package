import pprint as pp
import psicalc as pc

data1 = pc.durston_schema(pc.read_csv_file_format("hist-test.csv"), 1)
data2 = pc.durston_schema(pc.read_csv_file_format("test-hist-trunc.csv"), 1)
labels = ['A', 'B']
msa = pc.merge_sequences([data1, data2], labels)
result = pc.find_clusters(1, msa, "pairwise", 0.0)

pp.pprint(result)

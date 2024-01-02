#!/bin/env python3
"""
Psi-calc is an algorithm for clustering protein multiple sequence alignments (MSAs)
utilizing normalized mutual information.

    Copyright (C) 2020 Thomas Townsley, MSSE, Joe Deweese, PhD., Kirk Durston, PhD.,
    Timothy Wallace, PhD, Salvador Cordova, PhD et. al.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact: thomas@mandosoft.dev, joe.deweese@lipscomb.edu, kirkdurston@gmail.com
"""
import sys
import re
import time
import csv
import warnings
import pandas as pd
import numpy as np
from itertools import combinations
from .nmi import entropy
from .nmi import normalized_mutual_info_score
from .nmi_cache import NmiCache

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

nmi_cache = NmiCache(normalized_mutual_info_score)


def select_subset(c_list: list, s: int):
    """Selects the width of the sample subset from the MSA"""
    each_nth_col = c_list[::s]
    return each_nth_col


def durston_schema(df: pd.DataFrame, value: int) -> pd.DataFrame:
    """Labels index based on first value given. For example making the value 3
    will label the columns 3-N."""
    df.columns = range(len(df.columns))
    label_val = value
    df = df.rename(columns=lambda x: int(x) + label_val)

    return df


# noinspection PyUnresolvedReferences
def deweese_schema(df: pd.DataFrame, pattern='^-') -> pd.DataFrame:
    """Labels data based on the range on the first row of the MSA.
    For example, if the first row is labelled TOP2 YEAST/59-205, then all
    columns with individuals in the first row are kept and labeled 59-205.

    As an option, you can supply a different regex if the MSA uses different
    symbols to represent insertions."""

    try:
        first_row_ix = df.index[0]
        if '(' in first_row_ix:
            ix_label = first_row_ix.rsplit('(', 1)
        elif ':' in first_row_ix:
            ix_label = first_row_ix.rsplit(':', 1)
        else:
            ix_label = first_row_ix.rsplit('/', 1)
        ix_label = ix_label[1]
        ix_label = ix_label.rsplit('-', 1)
        df_label = int(ix_label[0])
        column_lab_dict = dict()

        if pattern == 'None':
            for i in df:
                if df[i].iloc[0] is not None:
                    column_lab_dict[df.columns[i]] = df_label
                    df_label += 1
                else:
                    column_lab_dict[df.columns[i]] = ''
        else:
            for i in df:
                if not re.search(pattern, df[i].iloc[0]):
                    column_lab_dict[df.columns[i]] = df_label
                    df_label += 1
                else:
                    column_lab_dict[df.columns[i]] = ''

        df = df.rename(columns=column_lab_dict)
        if '' in column_lab_dict.values():
            df = df.drop(columns=[''])
        df = df.rename(columns=lambda x: int(x))

    except IndexError or KeyError:
        sys.exit(1)

    return df


def check_for_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate sequences based on sequence label only."""
    raw_data = len(df.index)
    df = df[~df.index.duplicated(keep='first')]
    dd = len(df.index)
    if raw_data != dd:
        print((raw_data - dd), " rows removed due to duplicate SEQUENCE_ID")

    return df


def read_txt_file_format(file) -> pd.DataFrame:
    """Reads FASTA files or text file-based MSAs into a dataframe."""
    id, seq = [], []
    with open(file, "r") as fasta_file:
        for line in fasta_file:
            if line.startswith(">"):
                id.append(line.strip("\n").strip(">"))
            elif line:
                seq.append([*line.strip("\n")])
        res = list(zip(id, seq))
        fasta_file.close()
    nucs_dict = dict(res)

    df = pd.DataFrame.from_dict(nucs_dict, orient='index')
    df = df.replace({'.': '-'})
    df.index.name = 'SEQUENCE_ID'
    df = check_for_duplicates(df)
    df.fillna('?', inplace=True)

    return df


def read_csv_file_format(file) -> pd.DataFrame:
    """Reads CSV MSAs into a dataframe."""
    df = pd.read_csv(file, encoding='utf-8', engine='c', header=None)
    df = df.rename(columns={df.columns[0]: 'SEQUENCE_ID'})
    df = df.set_index('SEQUENCE_ID', drop=True)
    df = check_for_duplicates(df)
    df.fillna('?', inplace=True)

    if df.index[0] == "Domain: Data":
        df.drop(index=["Domain: Data"], inplace=True)
    df.columns = range(len(df.columns))

    return df


def return_sr_mode(msa: np.ndarray, m_map: dict, c: list, c_dict: dict, list_store: list, k) -> tuple:
    """Calculates the sr_mode and new mode of a cluster and returns both."""
    cc = len(list(combinations(c, 2)))
    if len(c) < 2:
        sr_mode, new_mode = None, None
        return sr_mode, new_mode
    elif len(c) == 2:
        i, j = m_map.get(c[0]), m_map.get(c[1])
        max_sum = nmi_cache.get(i, j, msa)

        new_mode = c
        """
        if max_sum > 1.0:
            print("\nERROR: Score returned greater than 1.0: ")
            print(c, max_sum)
            exit(1)
        """
    else:
        A = [[column, []] for column in range(len(c))]
        D = len(A)
        shift = 0
        for loc, i in enumerate(c):
            for location, j in enumerate(c):
                if location != loc and location > shift:
                    l, r = m_map.get(i), m_map.get(j)
                    A[loc][1].append(nmi_cache.get(l, r, msa))
            t = loc
            q = loc + 1
            if q != D:
                while q != D:
                    A[q][1].append(A[loc][1][t])
                    q += 1
                    t += 1
                shift += 1

        mode_map = np.array([np.sum(v) for (k, v) in A], dtype='d')
        mode_loc = int(np.argmax(mode_map))
        max_sum = np.amax(mode_map)
        new_mode = return_new_mode(mode_loc, c)

    sr_mode = max_sum / cc
    if k == "pairwise" or k == "top-pairwise":
        list_store.append([sr_mode, new_mode])

    if k != "pairwise" or k == "pairwise_only":
        c_dict[tuple(sorted(tuple(c)))] = [round(sr_mode, 6), k]

    return sr_mode, new_mode


def return_new_mode(location: int, c: list) -> list:
    """Calculates the new mode of a cluster and returns the updated cluster.
    The mode of a cluster is the left-most column in the dataframe."""
    mode = c[location]
    c.remove(mode)
    c.append(mode)
    c.reverse()
    return c


def check_intersections(agg: list) -> list:
    """Finds clusters with overlapping attributes and takes the one
    with the highest sr_mode value."""
    idx = 0
    while idx < len(agg):
        iex = 0

        while iex < len(agg):

            set1_sr_mode, set2_sr_mode = agg[idx][0], agg[iex][0]
            set1, set2 = set(agg[idx][1]), set(agg[iex][1])

            # check for intersections
            if set1 != set2 and set1.intersection(set2):
                if set1_sr_mode > set2_sr_mode:
                    del agg[iex]
                    iex -= 1
                else:
                    del agg[idx]
                    idx -= 1
                    break

            # check for permutations
            elif set1 == set2 and agg[idx][1][0] != agg[iex][1][0]:
                del agg[iex]
                iex -= 1
            else:
                iex += 1

        idx += 1

    return [j for x, j in agg]


def write_output_data(spread: int, c_dict: dict):
    """Writes the CSV output file"""
    filename = "data_out_width" + str(spread) + ".csv"
    a_file = open(filename, "w")
    writer = csv.writer(a_file)
    writer.writerow(["Cluster", "Sr_mode", "Discovered"])
    for key, value in c_dict.items():
        val1, val2 = value
        writer.writerow([key, val1, val2])
    a_file.close()

    return filename


def get_unique_elements(df: pd.DataFrame) -> np.ndarray:
    """Returns all unique elements found in a multiple
    sequence alignment."""
    U = np.array([])
    for name, seq in df.iteritems():
        U = np.append(U, seq.unique())

    return np.unique(U)


def rectify_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects symbols in the multiple sequence alignment."""
    alpha = get_unique_elements(df)
    pattern = '[-#?.]'
    allowed = '[ACDEFGHIKLMNPQRSTVWY]'
    invalid = '[BJOUXZ]|[bjouxz]'
    for symbol in alpha:
        if re.search(allowed, symbol):
            pass
        else:
            if re.search(invalid, symbol):
                df.replace(symbol, '-', inplace=True)
            elif symbol.islower() and len(symbol) == 1:
                df.replace(symbol, symbol.upper(), inplace=True)
            elif re.search(pattern, symbol):
                df.replace(symbol, '-', inplace=True)
            elif len(symbol) > 1:
                df.replace(symbol, '-', inplace=True)
            else:
                print("FAILED: Bad symbol in MSA: ", symbol)
                exit(1)
    print("\nBad symbols were discovered and replaced. New symbols:\n", get_unique_elements(df))
    return df


def encode_msa(df: pd.DataFrame) -> np.ndarray:
    """Naive integer encoding. Encodes amino acid values with
    unique integers and returns a numpy matrix"""
    ent = 0
    alpha = get_unique_elements(df)
    if alpha.size > 21:
        df = rectify_sequences(df)
        alpha = get_unique_elements(df)
    for symbol in alpha:
        if symbol == '-':
            df.replace(symbol, 0, inplace=True)
        else:
            ent += 1
            df.replace(symbol, ent, inplace=True)
    if ent > 21:
        print("FAILED: Too many labels encoded: ", ent)
        exit(1)
    print("\nNumber of labels encoded: ", (ent + 1), " including gaps.")
    np_matrix = df.to_numpy()

    return np_matrix


def remove_high_insertion_sites(df: pd.DataFrame, value: int) -> pd.DataFrame:
    """Useful for removing high insertion columns from MSA data.
    Returns the pandas dataframe with high insertion attributes removed.

    Provide percentage value to remove as a whole number, i.e. 15 % == 15
    """
    df.replace({'[-#?.]': None}, regex=True, inplace=True)
    index_len = len(df.index)
    null_val = float(value) / 100
    labels_to_delete = []

    def series_remove_insertions(x):
        non_nulls = x.count()
        info_amount = non_nulls / index_len
        if info_amount < null_val:
            labels_to_delete.append(x.name)

    # noinspection PyTypeChecker
    df.apply(series_remove_insertions, axis=0)
    df.drop(labels_to_delete, axis=1, inplace=True)
    df.replace({None: '-'}, inplace=True)

    return df


def return_dict_state():
    """Useful for returning the values of the global cluster dictionary
    at an intermediate stage. For example, a user may want to end a long
    running process and grab the available data prior to exit."""

    global halt
    if halt is False:
        halt = True

    return


def signal_halt() -> bool:
    """Gets the state of the halt global variable. Allows for termination
    of long running process in a safe manner."""

    global halt
    if halt:
        halt = False
        return True

    return False


def calculate_time(start):
    """Calculate time taken for program execution."""
    print("\n\n--- took " + str(round((time.time() - start), 3)) + " seconds ---")

    return


def merge_sequences(data: [pd.DataFrame], labels: [str]) -> pd.DataFrame:
    """
    Combine `data` into one big MSA and fill gaps due to dimension mismatch with
    '-'. Ignores mismatched indices. Loses row indices
    """

    # Rename columns with provided names unless there's only one MSA and no name
    # is provided.
    if len(labels) > 0:
        assert len(data) == len(labels), "MSA names != MSA columns"
        for i in range(len(data)):
            data[i] = data[i].rename(columns=lambda x: labels[i] + str(x))

    data = [d.reset_index(drop=True) for d in data]
    data = pd.concat(data, axis=1)
    data = data.fillna('-')
    return data


def prepare_data(data: pd.DataFrame) -> (np.ndarray, dict):
    """
    Prepares MSA data for clustering. Returns the encoded MSA and a mapping of the
    column indices to labels.
    """

    print("\nEncoding MSA(s)...")
    msa = encode_msa(data)

    # Create a mapping of column indices to names
    col_names = data.columns.tolist()
    col_indices = list(range(msa.shape[1]))
    col_map = dict(zip(col_indices, col_names))

    return msa, col_map


def filter_entropy(msa: np.ndarray, column_map: dict, e: float) -> (np.ndarray, list, list):
    """
    Filters all columns of `msa` with entropy lower than `e`. Returns the filtered msa and lists
    of included and filtered msa column names.
    """

    num_columns = msa.shape[1]

    # Create lists of low entropy names and column indices, plus interesting msa column names
    low_entropy_sites = []
    low_entropy_columns = []
    msa_names = []
    for idx in range(num_columns):
        if entropy(msa[:, idx]) < e:
            low_entropy_columns.append(idx)
            low_entropy_sites.append(column_map[idx])
        else:
            msa_names.append(column_map[idx])

    # Create a new matrix without the low entropy regions
    msa = np.delete(msa, low_entropy_columns, axis=1)

    return msa, msa_names, low_entropy_sites



def find_clusters(spread: int, msa: pd.DataFrame, k="pairwise", e=0.0) -> dict:
    """
    Discovers cluster sites with high shared normalized mutual information.
    Provide a dataframe and a sample spread-width. Returns a dictionary.

    The variable k by default is set to 'pairwise' to look for pairwise clusters
    and aggregate them prior to running Phase 2. Users may also set k to 'pairwise_only'
    if they only want to output pairwise clusters without aggregating them, in which
    case the program will halt once all pairwise clusters are found. This may be useful
    when comparing to methods like DCA.

    By default "e" or entropy is set to 0 but may be adjusted to exclude low entropy sites from
    being run in the program.
    """

    global halt
    halt = False
    csv_dict = dict()
    start_time = time.time()
    hash_list = list()

    # Map labels to columns
    msa, column_map = prepare_data(msa.copy(deep=True))
    csv_dict["column_map"] = column_map

    # Filter low entropy sites
    num_msa, msa_attrs, low_entropy_sites = filter_entropy(msa, column_map, e)
    print("\nNumber of low entropy regions excluded: ", len(low_entropy_sites))
    csv_dict["low_entropy_sites"] = low_entropy_sites

    msa_map = {k: v for v, k in enumerate(msa_attrs)}
    subset = select_subset(msa_attrs, spread)
    subset_list = [[z] for z in subset]

    print("\nLooking for strong pairwise clusters...")
    for item, each in enumerate(subset_list):
        max_rii, best_cluster = 0.0, None
        for location, cluster in enumerate(msa_attrs):
            cluster_mode = msa_map.get(cluster)
            subset_mode = msa_map.get(each[0])

            if subset_mode != cluster_mode:
                rii = nmi_cache.get(subset_mode, cluster_mode, num_msa)
                if rii > max_rii:
                    max_rii, best_cluster = rii, location
        if best_cluster is None:
            continue
        else:
            subset_list[item].append(msa_attrs[best_cluster])
            print("pair located: ", subset_list[item])

    pair_list = [x for x in subset_list if len(x) > 1]

    for cluster in pair_list:
        return_sr_mode(num_msa, msa_map, cluster, csv_dict, hash_list, k)

    # Sort list by Sr_Mode and generate sorted list of labels
    # May also return pairwise data if no strong sites are found
    sorted_list = sorted(hash_list, key=lambda x: x[0], reverse=True)
    out_list = [x for x in sorted_list if x[0] >= 0.10]

    # Used only when user selects to obtain pairwise clusters only
    if k == "pairwise_only" or len(out_list) == 0:
        calculate_time(start_time)
        return csv_dict

    # Check for attribute intersections between pairwise clusters
    final_df_set = check_intersections(out_list)

    unranked = list()
    k = "top-pairwise"
    for cluster in final_df_set:
        return_sr_mode(num_msa, msa_map, cluster, csv_dict, unranked, k)

    C = sorted(unranked, key=lambda x: x[0], reverse=True)

    for x, j in C:
        for col in j:
            try:
                msa_attrs.remove(col)
            except ValueError:
                print("\nFAILED: Tried to remove site location ", col, "but it was not found.\n"
                      "This is likely due to duplicates not being removed during check_intersections().")
                sys.exit()

    if len(msa_attrs) == 0:
        calculate_time(start_time)
        return csv_dict

    print("\nTop ranked clusters:")
    num = 0
    for each in C:
        num += 1
        print(num, ": ", each[1])

    # Sets the number of clusters we're actually iterating over
    R = len(C)
    for remaining in msa_attrs:
        C.append([0, [remaining]])
    num_clusters = len(C[0:R])
    cluster_halt = 0

    # Stage Two: Move through the pairs in the list and finds their best attribute
    try:
        while len(C) >= 1:

            if signal_halt():
                break

            print("\n --> Total Remaining ", len(C))

            i = 0
            while i < num_clusters:

                if signal_halt():
                    break

                location = None
                cluster_mode = msa_map.get(C[i][1][0])
                max_rii, best_cluster = 0.0, None

                for loc, entry in enumerate(C):
                    cluster = entry[1]
                    attr_mode = msa_map.get(cluster[0])
                    if cluster_mode != attr_mode:
                        rii = nmi_cache.get(attr_mode, cluster_mode, num_msa)
                        if rii > max_rii:
                            max_rii, best_cluster, location = rii, cluster, loc

                if best_cluster is None:
                    i += 1
                    cluster_halt += 1

                else:
                    cluster_halt = 0
                    C[i][1] = C[i][1] + best_cluster
                    C[i][0], C[i][1] = return_sr_mode(num_msa, msa_map, C[i][1], csv_dict, [], len(C))
                    C.pop(location)
                    i += 1
                    print("clusters: ", num_clusters,
                          " single attributes: ", (len(C) - num_clusters))
                    if len(C) == 1:
                        break
                    if location <= num_clusters:
                        num_clusters -= 1

            # Re-sort the list
            C = sorted(C, key=lambda x: x[0], reverse=True)

            if num_clusters <= 1 or cluster_halt >= num_clusters:
                break

    except KeyboardInterrupt:
        return_dict_state()

    calculate_time(start_time)
    halt = False

    return csv_dict

import numpy as np
import pandas as pd

# train_data shape: (num_triplet, 3), type: numpy, location: cpu
def head_tail_ratio(n_rel, train_data):
    stat = np.empty((n_rel, 2))
    train_data_for_stat = pd.DataFrame(train_data, columns=["head", "tail", "relation"])
    for relation in range(n_rel):
        head_count = len(
            train_data_for_stat[train_data_for_stat["relation"] == relation][["head"]].groupby(by=["head"]))
        tail_count = len(
            train_data_for_stat[train_data_for_stat["relation"] == relation][["tail"]].groupby(by=["tail"]))
        stat[relation] = np.array([head_count / (head_count + tail_count), tail_count / (head_count + tail_count)])
    return stat
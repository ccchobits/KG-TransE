import os
import numpy as np
import torch


def get_data(dataset_name, mode="train"):
    file_name = os.path.join("../data/raw", dataset_name, mode + "2id.txt")
    with open(file_name) as file:
        lines = file.read().strip().split("\n")
        n_triplets = int(lines[0])
        data = np.empty((n_triplets, 3), dtype=np.long)
        for i in range(1, len(lines)):
            line = lines[i]
            data[i - 1] = np.array([int(ids) for ids in line.split(" ")])
        assert n_triplets == len(data), "number of triplets is not correct."
        return n_triplets, torch.tensor(data)

import os
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from models.model_transE import TransE, TransE_norm
from utils.loader import get_data
from utils.processor import head_tail_ratio
from utils.writer import write_performance
from utils.logger import write_log

device = torch.device("cuda")

def bool_parser(s):
    if s not in {"True", "False"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"    

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--dataset', type=str, default='WN18')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--bs', type=int, default=2048, help="batch size")
parser.add_argument('--init_lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--bern', type=bool_parser, default=False, help="The strategy for sampling corrupt triplets. bern: bernoulli distribution.")
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--norm', type=int, default=2, help='[1 | 2]')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--dataset_path', type=str, default='../data/raw')
parser.add_argument('--mode', type=str, default='train', help='[prepro | train | test | infer]')
parser.add_argument('--log', type=bool_parser, default=True, help='logging or not')
parser.add_argument('--model', type=str, default="transE", help='The model for testing')
configs = parser.parse_args()



dataset_name = configs.dataset
bern = configs.bern
epochs = configs.epochs
batch_size = configs.bs
learning_rate = configs.init_lr
dim = configs.dim
margin = configs.margin
lr_decay = configs.lr_decay
norm = configs.norm

### load data
# train_data shape: (num_triplet, 3), type: torch.tensor, location: cpu
n_train, train_data = get_data(dataset_name=dataset_name, mode="train")
n_valid, valid_data = get_data(dataset_name=dataset_name, mode="valid")
n_test, test_data = get_data(dataset_name=dataset_name, mode="test")

n_ent = int(open(os.path.join("../data/raw", dataset_name, "entity2id.txt")).readline().strip())
n_rel = int(open(os.path.join("../data/raw", dataset_name, "relation2id.txt")).readline().strip())

### head to tail ratio statistic
if bern:
    stat = head_tail_ratio(n_rel, train_data.cpu().numpy())

### create model and optimizer
if configs.model == "transE":
    model = TransE(n_ent, n_rel, dim, margin, norm).to(device)
elif configs.model == "transE_norm":
    model = TransE_norm(n_ent, n_rel, dim, margin, norm).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### get corrupted samples
def get_neg_samples(pos_samples):
    size = len(pos_samples)
    new_ent = torch.randint(low=0, high=n_ent, size=(size,))
    if bern:
        head_or_tail = np.empty(size)
        rand = np.random.random(size)
        for i in range(size):
            if rand[i] < stat[pos_samples[i][2].item()][0]:
                head_or_tail[i] = 1
            else:
                head_or_tail[i] = 0
    else:
        head_or_tail = torch.randint(low=0, high=2, size=(size,))
    neg_samples = pos_samples.clone()
    for i in range(size):
        if head_or_tail[i] == 0:
            neg_samples.data[i][0] = new_ent[i]
        else:
            neg_samples.data[i][1] = new_ent[i]
    return neg_samples

### training the triplet in train_data
total_loss = 0

for epoch in range(1, epochs + 1):
    if epoch % 10 == 0:
        learning_rate /= lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffled_indices = torch.randperm(n_train)
    for i in range(0, n_train, batch_size):
        end = i + batch_size if i + batch_size <= n_train else n_train
        indice = shuffled_indices[i:end]
        pos_samples = train_data[indice]
        neg_samples = get_neg_samples(pos_samples)
        pos_samples, neg_samples = pos_samples.to(device), neg_samples.to(device)

        optimizer.zero_grad()
        loss = model(pos_samples, neg_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss

    if epoch % 20 == 0:
        print("epoch %d: lr: %.4f average loss per batch: %.4f" %
              (epoch, learning_rate, total_loss / (n_train // batch_size)), flush = True)
    total_loss = 0

### evaluate the triples in test_data
# triplet.type: torch.tensor, triplet.shape: (3, )
def triplet_to_string(triplet):
    # return format: "111 222 333"
    return " ".join([str(x) for x in triplet.tolist()])


all_triplets = set()
for dataset in [train_data, valid_data, test_data]:
    for triplet in dataset:
        all_triplets.add(triplet_to_string(triplet))


def rank(triplet):
    # head.shape, tail.shape, rel.shape: (batch_size,)
    head, tail, rel = model.ent_embedding(triplet[0]), model.ent_embedding(triplet[1]), model.rel_embedding(triplet[2])

    # predict tail
    new_triplet = triplet.clone()
    d = torch.norm(model.ent_embedding.weight.data - (head + rel), p=norm, dim=1)
    sorted_d_indices = d.sort(descending=False).indices
    tail_raw_ranking = np.where(sorted_d_indices.cpu().numpy() == triplet[1].item())[0][0].tolist() + 1
    tail_filtered_ranking = tail_raw_ranking
    for i in range(tail_raw_ranking - 1):
        new_triplet[1] = sorted_d_indices[i].item()
        if triplet_to_string(new_triplet) in all_triplets:
            tail_filtered_ranking -= 1

    # predict head
    new_triplet = triplet.clone()
    d = torch.norm(model.ent_embedding.weight.data - (tail - rel), p=norm, dim=1)
    sorted_d_indices = d.sort(descending=False).indices
    head_raw_ranking = np.where(sorted_d_indices.cpu().numpy() == triplet[0].item())[0][0].tolist() + 1
    head_filtered_ranking = head_raw_ranking
    for i in range(head_raw_ranking - 1):
        new_triplet[0] = sorted_d_indices[i].item()
        if triplet_to_string(new_triplet) in all_triplets:
            head_filtered_ranking -= 1

    return tail_raw_ranking, tail_filtered_ranking, head_raw_ranking, head_filtered_ranking

@torch.no_grad()
def evaluate():
    ranks = []
    for triplet in test_data:
        ranks.append(rank(triplet.to(device)))
    ranks = np.array(ranks)
    mean_rank = ranks.mean(axis=0, dtype=np.long)
    hit10 = np.sum(ranks <= 10, axis=0) / len(ranks)
    result = pd.DataFrame({"mean rank": mean_rank, "hit10": hit10},
                          index=["tail: raw ranking", "tail: filtered ranking", "head: raw ranking",
                                 "head: filtered ranking"])
    result["hit10"] = result["hit10"].apply(lambda x: "%.2f%%" % (x * 100))
    ranks = pd.DataFrame(ranks, columns = ["tail:raw", "tail:filtered", "head:raw", "head:filtered"])
    return ranks, result


model.eval()
ranks, result = evaluate()
write_performance(configs, result, "../scripts/asset/performance.result")

if configs.log:
    write_log(ranks)

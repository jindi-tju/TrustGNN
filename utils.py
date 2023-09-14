import numpy as np
import torch
import random
import os
from math import sqrt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_graph(dataset):
    edges = {}
    ecount = 0
    ncount = []
    edg = []
    lab = []
    with open('dataset/{}/{}-rating_label.txt'.format(dataset, dataset)) as dataset:
        for edge in dataset:
            ecount += 1
            ncount.append(edge.split()[0])
            ncount.append(edge.split()[1])
            edg.append(list(map(int, edge.split()[0:2])))
            lab.append(list(map(int, edge.split()[2:])))
    edges["labels"] = np.array(lab)
    edges["edges"] = np.array(edg)
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount))

    return edges


def eval_metric(scores, label):
    prediction = np.argmax(scores, axis=1).flatten()
    ac = accuracy_score(label, prediction)

    f1_micro = f1_score(label, prediction, average="micro")
    f1_macro = f1_score(label, prediction, average="macro")
    f1_weighted = f1_score(label, prediction, average="weighted")

    mae_convert = {0: 0.9, 1: 0.7, 2: 0.4, 3: 0.1}
    label_mae = [mae_convert[a] for a in label]
    prediction_mae = [mae_convert[a] for a in prediction]

    mae = mean_absolute_error(label_mae, prediction_mae)
    rmse = sqrt(mean_squared_error(label_mae, prediction_mae))

    return ac, f1_micro, f1_macro, f1_weighted, mae, rmse

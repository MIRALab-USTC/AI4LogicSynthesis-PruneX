

import pandas as pd
import numpy as np
import torch
import time
import json

from torch_geometric import loader
from utils.utils import GraphDataset
import utils.gcn_policy as gcn_policy


FEATURE_CSV = './features.csv'
TRUTHTABLE_CSV = './truth_table.csv'
DEPTH = 2
FANIN_NODES_TYPE = 'remove'
GCNPOLICY_CLASS = 'GCNPolicy'
BATCH_SIZE = 10240
POLICY_KWARGS = {
    'mean_max': 'mean',
    'emd_size': 128,
    'out_size': 2
}
with open('./configs/configs.json', 'r') as f:
    json_kwargs = json.load(f)
GCNMODEL = json_kwargs['GCNMODEL']
DEVICE = json_kwargs['DEVICE']
SEL_PERCENT = json_kwargs['SEL_PERCENT']
RANDOM = json_kwargs['RANDOM']
NORMALIZE = json_kwargs['NORMALIZE']

def _normalize(features):
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    normalized_features = (features - min_vals) / \
        (max_vals - min_vals + 1e-3)

    # print(f"debug log normalized features: {normalized_features}")
    return normalized_features

def _process_csv():
    # process collected csv
    data_frame = pd.read_csv(FEATURE_CSV)
    data_frame_truth_table = pd.read_csv(TRUTHTABLE_CSV)
    numpy_data = data_frame.to_numpy()
    numpy_data_truth_table = data_frame_truth_table.to_numpy()

    # get features current node
    features = numpy_data[:, 1:6].astype(np.int32)
    # get features truth table
    features_truth_table = numpy_data_truth_table[:, 1:]
    # get features x and label y
    features = np.concatenate((features, features_truth_table), axis=1)

    # first traverse id
    first_traverse_id = features[:, 4][0]

    # normalize features
    if NORMALIZE:
        features = _normalize(features)
    children = numpy_data[:, -2]
    parents = numpy_data[:, -1]

    return features, children, parents, first_traverse_id


def process_children(children, first_traverse_id):
    processed_children = []
    for i in range(children.shape[0]):
        processed_chhildren_i = []
        for ind in children[i].split("/"):
            try:
                processed_chhildren_i.append(int(ind))
            except:
                continue
            if DEPTH == 2:
                try:
                    actual_index = int(ind) - first_traverse_id
                    if actual_index < 0:
                        continue
                    for ind_2 in children[actual_index].split("/"):
                        try:
                            processed_chhildren_i.append(int(ind_2))
                        except:
                            continue
                except:
                    pass
                    #  print(f"found fanin nodes")
        processed_children.append(processed_chhildren_i)
        # print(f"{i} th children index: {processed_chhildren_i}")
    return processed_children


def get_children(children, i, first_traverse_id, features):
    children_indexes = children[i]
    children_features = []
    for id in children_indexes:
        index = id - first_traverse_id
        if index >= 0:
            children_feature = features[index:index+1, :]
            children_features.append(children_feature)
        else:
            if FANIN_NODES_TYPE == 'zero':
                children_feature = np.zeros((1, 69), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'one':
                children_feature = np.ones((1, 69), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'remove':
                continue
            children_features.append(children_feature)

    if len(children_features) == 0:
        children_features.append(np.ones((1, 69), dtype=np.int32))
    return np.vstack(children_features)


def get_edge_indexes(num_children):
    zero_row = np.zeros((1, num_children), dtype=np.int32)
    one_row = np.arange(num_children).astype(np.int32)
    one_row = np.expand_dims(one_row, axis=0)
    edge_indexes = np.vstack([zero_row, one_row])

    return edge_indexes


def process_data(features, children, first_traverse_id):
    samples = []
    for i in range(features.shape[0]):
        pivot_node_features = features[i:i+1, :]
        children_features = get_children(
            children, i, first_traverse_id, features)
        edge_indexes = get_edge_indexes(children_features.shape[0])
        label = np.array([0])
        sample = (pivot_node_features, edge_indexes, children_features, label)
        samples.append(sample)

    return samples


def evaluate_policy(graph_data_loader, graph_policy, first_traverse_id):
    predict_scores_list = []
    for batch in graph_data_loader:
        # print("evaluating data loader ..............")
        batch_pivot_node_features = batch.pivot_node_features.to(DEVICE)
        batch_children_node_features = batch.children_node_features.to(DEVICE)
        batch_edge_indexes = batch.edge_index.to(DEVICE)

        with torch.no_grad():
            predict_scores = graph_policy(
                batch_pivot_node_features,
                batch_edge_indexes,
                batch_children_node_features
            )
        predict_scores = predict_scores.cpu().detach().numpy()
        predict_scores_list.append(predict_scores)
        torch.cuda.empty_cache()
    predict_scores = np.vstack(predict_scores_list)
    predict_scores = np.mean(predict_scores, axis=1, keepdims=True)
    ascending_indexes = np.argsort(
        predict_scores.squeeze())  # default ascending order
    ascending_indexes += first_traverse_id
    total_num = predict_scores.shape[0]
    sel_num = int(total_num * SEL_PERCENT)
    sel_indexes = np.array(
        ascending_indexes[total_num-sel_num:], dtype=np.int32)
    # sel_indexes = sel_indexes[::-1]
    sel_indexes.sort()
    # print(sel_indexes)
    print(sel_indexes.shape[0])
    return sel_indexes


def load_graph_model():
    graph_policy = getattr(gcn_policy, GCNPOLICY_CLASS)(
        **POLICY_KWARGS).to(DEVICE)
    if GCNMODEL is not None:
        model_state_dict = torch.load(GCNMODEL, map_location=DEVICE)
        graph_policy.load_state_dict(model_state_dict)
    graph_policy.eval()

    return graph_policy

def get_random_indexes(num_samples, first_traverse_id):
    random_scores = [np.random.rand() for _ in range(num_samples)]
    ascending_indexes = np.argsort(random_scores)
    ascending_indexes += first_traverse_id
    sel_num = int(num_samples * SEL_PERCENT)
    sel_indexes = np.array(
        ascending_indexes[num_samples-sel_num:], dtype=np.int32)
    sel_indexes.sort()
    return sel_indexes

def online_inference():
    """
    # input: csv files and gnn models
    # output: the selected traverse ids
    """
    time_dict = {}
    time_dict["st"] = time.time()
    # First step: read csv process to numpy
    features, children, parents, first_traverse_id = _process_csv()
    time_dict["process_csv"] = time.time()
    # first_traverse_id = features[:, 4][0]
    num_samples = features.shape[0]
    processed_children = process_children(children, first_traverse_id)
    time_dict["process_children"] = time.time()
    # Second step: process numpy to graphs
    graph_samples = process_data(
        features, processed_children, first_traverse_id)
    time_dict["process_graph_samples"] = time.time()
    graph_dataset = GraphDataset(graph_samples)
    graph_data_loader = loader.DataLoader(
        graph_dataset,
        BATCH_SIZE,
        shuffle=False
    )
    # Third step: read models and inference
    graph_policy = load_graph_model()
    time_dict["load_graph_model"] = time.time()
    if RANDOM:
        sel_indexes = get_random_indexes(num_samples, first_traverse_id)
    else:
        sel_indexes = evaluate_policy(
            graph_data_loader, graph_policy, first_traverse_id)
    time_dict["policy inference"] = time.time()

    last_time = 0
    for k in time_dict.keys():
        if k == 'st':
            last_time = time_dict[k]
            continue
        print(f"time {k}: {time_dict[k]-last_time}")
        last_time = time_dict[k]
    # Fourth step: return
    return sel_indexes


if __name__ == "__main__":
    online_inference()

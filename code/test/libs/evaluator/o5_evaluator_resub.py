

import pandas as pd
import numpy as np
import torch
import time
import json

from torch_geometric import loader
from utils.utils import GraphDataset
import utils.gcn_policy as gcn_policy


FEATURE_CSV = './features_resub.csv'
Normalize = False
FANIN_NODES_TYPE = 'remove'
NUMFeatures = 20
GCNPOLICY_CLASS = 'GCNPolicy'
BATCH_SIZE = 10240
POLICY_KWARGS = {
    'mean_max': 'mean',
    'emd_size': 128,
    'out_size': 2,
    'num_pivot_node_features': 20,
    'num_children_node_features': 20
}
with open('./configs/configs.json', 'r') as f:
    json_kwargs = json.load(f)
GCNMODEL = json_kwargs['GCNMODEL']
DEVICE = json_kwargs['DEVICE']
SEL_PERCENT = json_kwargs['SEL_PERCENT']
RANDOM = json_kwargs['RANDOM']
Normalize = json_kwargs['NORMALIZE']
######## CSV TO NUMPY ########


def get_child_feature(features, numpy_data, i, first_traverse_id):
    # child_features = np.empty(
    #     (features.shape[0], features.shape[1]-2), dtype=np.int32)
    child_features = np.empty(
        (features.shape[0], features.shape[1]-2))
    # child_features = np.empty(
    #     (features.shape[0], features.shape[1]))
    first_id = first_traverse_id
    if i == 0:
        child_id = numpy_data[:, 8:9].astype(np.int32)
    elif i == 1:
        child_id = numpy_data[:, 9:10].astype(np.int32)
    child_index = child_id - first_id
    for j, ind in enumerate(child_index):
        if ind < 0:
            cur_child_features = np.ones((1, features.shape[1]))
        else:
            cur_child_features = features[ind]
            # print(f"debug log child shape: {cur_child_features.shape}")
        child_features[j] = cur_child_features[:, :-2]
        # child_features[j] = cur_child_features

    # print(f"debug log child features {i}: {child_features}")
    return child_features


def _normalize(features):
    max_vals = np.max(features, axis=0)
    min_vals = np.min(features, axis=0)
    normalized_features = (features - min_vals) / \
        (max_vals - min_vals + 1e-3)

    # print(f"debug log normalized features: {normalized_features}")
    return normalized_features


def get_div_ids(numpy_data):
    div_indexes = []
    first_id = numpy_data[0, 0]
    div_ids = numpy_data[:, 10:11]
    print(f"raw divisors len: {div_ids.shape}")
    for i, div_id in enumerate(div_ids):
        cur_div_index = []
        div_id = str(div_id)
        for id in div_id.split("/"):
            try:
                id = int(id) - first_id
                if i == id:
                    continue
                cur_div_index.append(id)
            except:
                continue
        div_indexes.append(cur_div_index)
    print(f"div indexes len: {len(div_indexes)}")
    return div_indexes


def _process_csv():
    # process collected csv
    data_frame = pd.read_csv(FEATURE_CSV)
    numpy_data = data_frame.to_numpy()

    # get features current node
    features = numpy_data[:, 0:8].astype(np.int32)
    first_traverse_id = features[0, 0]
    if Normalize:
        features = _normalize(features)
    # get children features
    child0features = get_child_feature(
        features, numpy_data, 0, first_traverse_id)
    child1features = get_child_feature(
        features, numpy_data, 1, first_traverse_id)
    # concat features
    features = np.concatenate(
        (features, child0features, child1features),
        axis=1
    )

    # get divs ids
    divs_ids = get_div_ids(numpy_data)
    return features, divs_ids, first_traverse_id
######## CSV TO NUMPY ########

######## NUMPY TO GRAPH ########


def get_children(div_ids, i, features):
    children_indexes = div_ids[i]
    children_features = []
    for index in children_indexes:
        if index >= 0:
            children_feature = features[index:index+1, :]
            children_features.append(children_feature)
        else:
            if FANIN_NODES_TYPE == 'zero':
                children_feature = np.zeros(
                    (1, NUMFeatures), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'one':
                children_feature = np.ones(
                    (1, NUMFeatures), dtype=np.int32)
            elif FANIN_NODES_TYPE == 'remove':
                continue
            children_features.append(children_feature)

    if len(children_features) == 0:
        children_features.append(np.ones((1, NUMFeatures), dtype=np.int32))
    return np.vstack(children_features)


def get_edge_indexes(num_children):
    zero_row = np.zeros((1, num_children), dtype=np.int32)
    one_row = np.arange(num_children).astype(np.int32)
    one_row = np.expand_dims(one_row, axis=0)
    edge_indexes = np.vstack([zero_row, one_row])

    return edge_indexes


def process_data(features, div_ids):
    samples = []
    for i in range(features.shape[0]):
        pivot_node_features = features[i:i+1, :]
        children_features = get_children(
            div_ids, i, features)
        edge_indexes = get_edge_indexes(children_features.shape[0])
        label = np.array([0])
        sample = (pivot_node_features, edge_indexes, children_features, label)
        samples.append(sample)

    return samples
######## NUMPY TO GRAPH ########


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
        model_state_dict = torch.load(GCNMODEL)
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
    features, divs_ids, first_traverse_id = _process_csv()
    time_dict["process_csv"] = time.time()
    num_samples = features.shape[0]
    # Second step: process numpy to graphs
    graph_samples = process_data(features, divs_ids)
    time_dict["process_graph_samples"] = time.time()
    graph_dataset = GraphDataset(graph_samples)
    graph_data_loader = loader.DataLoader(
        graph_dataset,
        BATCH_SIZE,
        shuffle=False
    )
    time_dict["load_graph_datasets"] = time.time()
    # Third step: read models and inference

    if RANDOM:
        sel_indexes = get_random_indexes(num_samples, first_traverse_id)
    else:
        graph_policy = load_graph_model()
        time_dict["load_graph_model"] = time.time()
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
    print(f"sel_indexes: {sel_indexes}")
    return sel_indexes


if __name__ == "__main__":
    online_inference()

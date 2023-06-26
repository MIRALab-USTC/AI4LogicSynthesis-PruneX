import time
import random
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from utils.const import LABEL_DICT

############## graph utils ##############


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, pivot_node_features, edge_indices, children_node_features, label):
        super().__init__()
        self.pivot_node_features = pivot_node_features
        self.edge_index = edge_indices
        self.children_node_features = children_node_features
        self.label = label

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.pivot_node_features.size(0)], [self.children_node_features.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, samples):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, index):
        sample = self.samples[index]
        pivot_node_features, edge_indices, children_node_features, label = sample
        pivot_node_features = torch.FloatTensor(pivot_node_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        children_node_features = torch.FloatTensor(children_node_features)
        label = torch.LongTensor(label.astype(np.int32))

        graph = BipartiteNodeData(
            pivot_node_features, edge_indices, children_node_features, label)
        graph.num_nodes = pivot_node_features.shape[0] + \
            children_node_features.shape[0]
        return graph


############## set global seeds ##############
def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time()) % 4096
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True
    return seed


def get_label_dict(npy_data_path):
    label_dict = {}
    cnt = 0
    for npy_file in os.listdir(npy_data_path):
        print(f"debug log npy file: {npy_file}")
        for k in LABEL_DICT.keys():
            if k in npy_file:
                label_dict[k] = cnt
                cnt += 1
                break

    return label_dict


if __name__ == '__main__':
    npy_data_path = './npy_data/v2/v2_data_train_log2'
    label_dict = get_label_dict(npy_data_path)

    print(label_dict)

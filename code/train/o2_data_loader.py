import copy
import os
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch_geometric import loader
from utils.utils import GraphDataset

NUMFeatures = 20


class ScoreDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ScoreDataLoader():
    def __init__(
        self,
        npy_data_path,
        batch_size=2048,
        sample_rate=2,
        sample_type='downsampling',
        operator_type='mfs2',
        normalize=False,
        train_type='train'
    ):
        self.npy_data_path = npy_data_path
        self.sample_rate = sample_rate
        self.sample_type = sample_type
        self.num_positive_samples = 0
        self.num_negative_samples = 0
        self.batch_size = batch_size
        self.operator_type = operator_type
        self.normalize = normalize
        self.train_type = train_type

        if self.sample_type == 'downsampling':
            process_data = self.process_data
        elif self.sample_type == 'upsampling':
            process_data = self.process_data_up_sample
        else:
            raise NotImplementedError
        self.dataset = {
            "x": None,
            "y": None,
            "default_order_features": None,
            "default_order_label": None
        }

        self.end_to_end_stats = None
        self.features = None
        self.labels = None

        self.load_data()

        self.std_dataset = ScoreDataset(self.dataset['x'], self.dataset['y'])
        self.num_samples = self.dataset['x'].shape[0]
        if self.train_type == 'train':
            self.batch_data_generator = DataLoader(
                self.std_dataset,
                batch_size=self.batch_size, shuffle=True)
        elif self.train_type == 'test':
            if self.batch_size >= self.num_samples:
                self.test_data_loader = DataLoader(
                    self.std_dataset,
                    batch_size=self.num_samples, shuffle=False)
            else:
                self.test_data_loader = DataLoader(
                    self.std_dataset,
                    batch_size=self.batch_size, shuffle=False)

    def _normalize(self):
        max_vals = np.max(self.features, axis=0)
        min_vals = np.min(self.features, axis=0)
        normalized_features = (self.features - min_vals) / \
            (max_vals - min_vals + 1e-3)
        self.features = normalized_features
        # print(f"debug log normalized features: {normalized_features}")
        # return normalized_features
    
    def load_data(self):
        # load data
        if self.npy_data_path.endswith('.npy'):
            npy_file_path = self.npy_data_path
            npy_data = np.load(npy_file_path, allow_pickle=True).item()
            if self.operator_type == 'mfs2':
                self.features = npy_data['features_list'][0][:,:5]
            elif self.operator_type == 'resub':
                self.features = npy_data['features_list'][0][:,:12]
            elif self.operator_type == 'original':
                self.features = npy_data['features_list'][0]

            self.labels = npy_data['labels_list'][0]
        else:
            features = []
            labels = []
            for npy_file in os.listdir(self.npy_data_path):
                npy_file_path = os.path.join(self.npy_data_path, npy_file)
                npy_data = np.load(npy_file_path, allow_pickle=True).item()
                if self.operator_type == 'mfs2':
                    features.append(npy_data['features_list'][0][:,:5])
                elif self.operator_type == 'resub':
                    features.append(npy_data['features_list'][0][:,:12])
                elif self.operator_type == 'original':
                    features.append(npy_data['features_list'][0])
                labels.append(npy_data['labels_list'][0])
                # print(f"cur npy: {npy_file}, {npy_data['features_list'][0][:,:5]}")
            self.features = np.vstack(features)
            self.labels = np.vstack(labels)

        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)[0]
        negative_indexes = np.nonzero(self.labels == 0)[0]
        # upsampling positive samples
        num_positive_sample = len(positive_indexes)
        num_negative_sample = len(negative_indexes)
        self.num_positive_samples += num_positive_sample
        self.num_negative_samples += num_negative_sample

        # normalize
        if self.normalize:
            self._normalize()
        self.dataset['default_order_features'] = self.features
        self.dataset['default_order_label'] = self.labels
        # x y training data
        self.dataset['x'] = self.features
        self.dataset['y'] = self.labels

    def process_data_up_sample(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)
        print(positive_indexes)
        positive_x = self.features[positive_indexes[0]]
        positive_y = self.labels[positive_indexes[0]]
        print(f"postive x shape: {positive_x.shape}")
        print(f"postive y shape: {positive_y.shape}")

        negative_indexes = np.nonzero(self.labels == 0)
        negative_x = self.features[negative_indexes[0]]
        negative_y = self.labels[negative_indexes[0]]
        print(f"negative x shape: {negative_x.shape}")
        print(f"negative y shape: {negative_y.shape}")

        # upsampling positive samples
        num_positive_sample = positive_x.shape[0]
        num_negative_sample = negative_x.shape[0]
        upsample_rate = int(num_negative_sample / num_positive_sample)
        print(f"before positive x: {positive_x}")
        print(f"before positive x: {positive_y}")
        positive_x = np.repeat(positive_x, upsample_rate, axis=0)
        positive_y = np.repeat(positive_y, upsample_rate, axis=0)
        print(f"after positive x: {positive_x}")
        print(f"after positive x: {positive_y}")
        print(f"augmented positive x shape: {positive_x.shape}")
        print(f"augmented positive y shape: {positive_y.shape}")

        x = np.concatenate((positive_x, negative_x))
        y = np.concatenate((positive_y, negative_y))

        # np.random.shuffle(x)
        # np.random.shuffle(y)

        self.dataset['x'] = x
        self.dataset['y'] = y

        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")

    def process_data(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)
        print(positive_indexes)
        positive_x = self.features[positive_indexes[0]]
        positive_y = self.labels[positive_indexes[0]]
        print(f"postive x shape: {positive_x.shape}")
        print(f"postive y shape: {positive_y.shape}")

        negative_indexes = np.nonzero(self.labels == 0)
        negative_x = self.features[negative_indexes[0]]
        negative_y = self.labels[negative_indexes[0]]
        print(f"negative x shape: {negative_x.shape}")
        print(f"negative y shape: {negative_y.shape}")

        # downsampling negative samples
        num_positive_sample = positive_x.shape[0]
        num_negative_sample = negative_x.shape[0]
        down_sample_num_negative_sample = int(
            num_positive_sample * self.sample_rate)
        indexes = np.arange(num_negative_sample)
        np.random.shuffle(indexes)
        sample_indexes = indexes[:down_sample_num_negative_sample]
        negative_x = negative_x[sample_indexes]
        negative_y = negative_y[sample_indexes]
        print(f"sample negative x shape: {negative_x.shape}")
        print(f"sample negative y shape: {negative_y.shape}")

        x = np.concatenate((positive_x, negative_x))
        y = np.concatenate((positive_y, negative_y))

        # np.random.shuffle(x)
        # np.random.shuffle(y)

        self.dataset['x'] = x
        self.dataset['y'] = y

        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")

    # def batch_data_generator(self, batch_size):
    #     total_num = self.dataset['y'].shape[0]
    #     sampler = BatchSampler(
    #         SubsetRandomSampler(range(total_num)),
    #         batch_size,
    #         drop_last=False)
    #     for idxes in sampler:
    #         # print(f"idxes: {idxes}")
    #         sample_x = self.dataset['x'][idxes]
    #         sample_y = self.dataset['y'][idxes]

    #         yield sample_x, sample_y


class GraphDataLoader(DataLoader):
    def __init__(
        self,
        npy_data_path=None,
        save_dir=None,
        processed_npy_path=None,
        sample_rate=2,
        sample_bool=True,
        train_type='train',
        sample_type='upsampling',
        load_type='default_order',
        batch_size=128,
        max_batch_size=10240,
        depth=2,
        # debug
        debug_log=False,
        # domain attribute
        label_dict=None,
        # fanin_nodes
        fanin_nodes_type='zero',
        normalize=False
    ):
        self.npy_data_path = npy_data_path
        self.sample_rate = sample_rate
        self.sample_type = sample_type
        self.load_type = load_type
        self.sample_bool = sample_bool
        self.train_type = train_type
        self.save_dir = save_dir
        self.depth = depth
        self.debug_log = debug_log
        self.label_dict = label_dict
        self.fanin_nodes_type = fanin_nodes_type
        self.max_batch_size = max_batch_size
        self.normalize = normalize

        if self.sample_type == 'downsampling':
            process_data = self.process_data
        elif self.sample_type == 'upsampling':
            process_data = self.process_data_up_sample
        else:
            raise NotImplementedError
        self.dataset = {
            "samples": [],
            "default_order_features": None,
            "default_order_label": None,
            "default_order_children_features": None,
            "default_traverse_id": None,
            "first_traverse_id": None
        }

        self.end_to_end_stats = None
        self.features = None
        self.labels = None
        self.children = None
        # num samples
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_negative_samples = 0
        samples_list = []
        if processed_npy_path is None:
            if npy_data_path.endswith('.npy'):
                npy_file_path = npy_data_path
                self.load_data(npy_file_path)
                samples, bool_con = process_data()
                samples_list.extend(samples)
            else:
                for npy_file in os.listdir(npy_data_path):
                    npy_file_path = os.path.join(npy_data_path, npy_file)
                    print(f"cur npy: {npy_file}")
                    self.load_data(npy_file_path)
                    samples, bool_con = process_data()
                    if bool_con:
                        continue
                    samples_list.extend(samples)
            # save processed samples
            # np.save(
            #     f"../../../npy_data/v2/{save_dir}/graph_samples_train_type_{train_type}_sample_bool_{self.sample_bool}.npy", samples_list)
        else:
            samples_list = np.load(processed_npy_path, allow_pickle=True)

        self.graph_dataset = GraphDataset(samples_list)
        if self.train_type == 'train':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
            else:
                sampler = BatchSampler(
                    SubsetRandomSampler(range(total_num)),
                    self.max_batch_size,
                    drop_last=False)
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, batch_sampler=sampler)
        elif self.train_type == 'test':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
            else:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.max_batch_size, shuffle=False)

    def _get_label(self, npy_file_path):
        for k in self.label_dict.keys():
            if k in npy_file_path:
                label_value = self.label_dict[k]
        return label_value

    def _normalize(self):
        max_vals = np.max(self.features, axis=0)
        min_vals = np.min(self.features, axis=0)
        normalized_features = (self.features - min_vals) / \
            (max_vals - min_vals + 1e-3)
        self.features = normalized_features
        
    def load_data(self, npy_file_path):
        npy_data = np.load(npy_file_path, allow_pickle=True).item()
        self.end_to_end_stats = npy_data['end_to_end_stats']
        self.dataset['default_order_label'] = npy_data['labels_list'][0]
        self.dataset['default_traverse_id'] = list(
            npy_data['features_list'][0][:, 4])
        self.dataset['default_order_children_features'] = self.process_children(
            [npy_data['children'][0]])
        self.dataset['first_traverse_id'] = self.dataset['default_traverse_id'][0]
        
        self.features = npy_data['features_list'][0]
        if self.normalize:
            self._normalize()
        self.dataset['default_order_features'] = self.features

        if self.label_dict is None:
            self.labels = npy_data['labels_list'][0]
        else:
            label_value = self._get_label(npy_file_path)
            self.labels = np.ones(
                (self.features.shape[0], 1)) * label_value
            self.labels = self.labels.astype(np.int64)
        self.children = self.process_children([npy_data['children'][0]])
        self.num_samples = self.labels.shape[0]
        if self.debug_log:
            print(f"debug log features: {self.features}")
            print(f"debug log labels: {self.labels}")
            print(
                f"debug log default_traverse_id: {self.dataset['default_traverse_id']}")
            print(f"debug log children: {self.children}")
            print(f"debug log npy file: {npy_file_path}")
            print(
                f"debug log first_traverse_id: {self.dataset['first_traverse_id']}")

    def process_children(self, children):
        processed_children = []
        for child_list in children:
            for i in range(child_list.shape[0]):
                processed_chhildren_i = []
                for ind in child_list[i].split("/"):
                    try:
                        processed_chhildren_i.append(int(ind))
                    except:
                        continue
                    if self.depth == 2:
                        try:
                            # actual_index = self.dataset['default_traverse_id'].index(
                            #     int(ind))
                            actual_index = int(ind) - self.dataset['first_traverse_id']
                            # print(f"found actual index: {actual_index}")
                            for ind_2 in child_list[actual_index].split("/"):
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

    def get_children(self, i):
        children_indexes = self.children[i]
        children_features = []
        for id in children_indexes:
            index = id - self.dataset['first_traverse_id']
            if index >= 0:
                # index = self.dataset['default_traverse_id'].index(id)
                children_feature = self.dataset['default_order_features'][index:index+1, :]
                children_features.append(children_feature)
            else:
                if self.fanin_nodes_type == 'zero':
                    children_feature = np.zeros((1, 69), dtype=np.int32)
                elif self.fanin_nodes_type == 'one':
                    children_feature = np.ones((1, 69), dtype=np.int32)
                elif self.fanin_nodes_type == 'remove':
                    continue
                children_features.append(children_feature)

            # try:
            #     index = self.dataset['default_traverse_id'].index(id)
            #     children_feature = self.dataset['default_order_features'][index:index+1,:]
            #     children_features.append(children_feature)
            # except:
            #     # print(f"current node ones fanin!!!!")
            #     if self.fanin_nodes_type == 'zero':
            #         children_feature = np.zeros((1,69), dtype=np.int32)
            #     elif self.fanin_nodes_type == 'one':
            #         children_feature = np.ones((1,69), dtype=np.int32)
            #     elif self.fanin_nodes_type == 'remove':
            #         continue
            #     children_features.append(children_feature)

        if len(children_features) == 0:
            children_features.append(np.ones((1, 69), dtype=np.int32))
        return np.vstack(children_features)

    def get_edge_indexes(self, num_children):
        zero_row = np.zeros((1, num_children), dtype=np.int32)
        one_row = np.arange(num_children).astype(np.int32)
        one_row = np.expand_dims(one_row, axis=0)
        edge_indexes = np.vstack([zero_row, one_row])

        return edge_indexes

    def process_data_up_sample(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)[0]
        negative_indexes = np.nonzero(self.labels == 0)[0]
        # upsampling positive samples
        num_positive_sample = len(positive_indexes)
        num_negative_sample = len(negative_indexes)
        self.num_positive_samples += num_positive_sample
        self.num_negative_samples += num_negative_sample
        if num_positive_sample <= 0:
            return [], True
        upsample_rate = int(num_negative_sample / num_positive_sample)

        print(f"children shape: {len(self.children)}")
        print(f"total samples: {num_positive_sample+num_negative_sample}")
        print(f"num_positive_samples: {num_positive_sample}")
        print(f"num_negative_samples: {num_negative_sample}")
        samples = []
        for i in range(num_positive_sample+num_negative_sample):
            pivot_node_features = self.features[i:i+1, :]
            label = self.labels[i:i+1, :]
            children_features = self.get_children(i)
            edge_indexes = self.get_edge_indexes(children_features.shape[0])
            sample = (pivot_node_features, edge_indexes,
                      children_features, label)
            samples.append(sample)
            if self.sample_bool:
                if i in positive_indexes:
                    # repeat positive samples
                    for _ in range(upsample_rate):
                        samples.append(sample)
            if self.debug_log:
                print(
                    f"debug log sample {i}, pivot_node_features: {pivot_node_features}")
                print(f"debug log sample {i}, label: {label}")
                print(
                    f"debug log sample {i}, children_features: {children_features}")
                print(f"debug log sample {i}, edge_indexes: {edge_indexes}")

        self.dataset['samples'] = samples
        return samples, False


class GraphResubDataLoader(GraphDataLoader):
    def __init__(
        self,
        npy_data_path=None,
        save_dir=None,
        processed_npy_path=None,
        train_type='train',
        batch_size=128,
        max_batch_size=10240,
        # debug
        debug_log=False,
        # domain attribute
        label_dict=None,
        # fanin_nodes
        fanin_nodes_type='zero',
        normalize=False
    ):
        self.npy_data_path = npy_data_path
        self.train_type = train_type
        self.save_dir = save_dir
        self.debug_log = debug_log
        self.label_dict = label_dict
        self.fanin_nodes_type = fanin_nodes_type
        self.max_batch_size = max_batch_size
        self.normalize = normalize

        # datasets
        self.end_to_end_stats = None
        self.features = None
        self.labels = None
        self.divisors = None
        self.first_traverse_id = None
        # num samples
        self.num_samples = 0
        self.num_positive_samples = 0
        self.num_negative_samples = 0
        samples_list = []
        if processed_npy_path is None:
            if npy_data_path.endswith('.npy'):
                npy_file_path = npy_data_path
                self.load_data(npy_file_path)
                samples, bool_con = self.process_data()
                if bool_con:
                    print(
                        f"warning! current npy data {npy_file_path} zero positive sample")
                samples_list.extend(samples)
            else:
                for npy_file in os.listdir(npy_data_path):
                    print(f"current npy file: {npy_file}")
                    if npy_file.endswith('.npy'):
                        npy_file_path = os.path.join(npy_data_path, npy_file)
                        self.load_data(npy_file_path)
                        samples, bool_con = self.process_data()
                        print(f"bool continuous: {bool_con}")
                        if bool_con:
                            continue
                        samples_list.extend(samples)
            # save processed samples
            # np.save(
            #     f"../../../npy_data/v2/{save_dir}/graph_samples_train_type_{train_type}_sample_bool_{self.sample_bool}.npy", samples_list)
        else:
            samples_list = np.load(processed_npy_path, allow_pickle=True)

        self.graph_dataset = GraphDataset(samples_list)
        if self.train_type == 'train':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
            else:
                sampler = BatchSampler(
                    SubsetRandomSampler(range(total_num)),
                    self.max_batch_size,
                    drop_last=False)
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, batch_sampler=sampler)
        elif self.train_type == 'test':
            total_num = len(samples_list)
            if total_num <= self.max_batch_size:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.num_samples, shuffle=False)
            else:
                self.graph_data_loader = loader.DataLoader(
                    self.graph_dataset, self.max_batch_size, shuffle=False)

    def _normalize(self):
        max_vals = np.max(self.features, axis=0)
        min_vals = np.min(self.features, axis=0)
        normalized_features = (self.features - min_vals) / \
            (max_vals - min_vals + 1e-3)
        self.features = normalized_features

    def load_data(self, npy_file_path):
        npy_data = np.load(npy_file_path, allow_pickle=True).item()
        self.end_to_end_stats = npy_data['end_to_end_stats']
        self.features = npy_data['features_list'][0]
        self.first_traverse_id = self.features[0, 0]
        if self.normalize:
            self._normalize()
        if self.label_dict is None:
            self.labels = npy_data['labels_list'][0]
        else:
            label_value = self._get_label(npy_file_path)
            self.labels = np.ones(
                (self.features.shape[0], 1)) * label_value
            self.labels = self.labels.astype(np.int32)
        self.divisors = npy_data['divs_list'][0]
        self.num_samples = self.labels.shape[0]

        if self.debug_log:
            print(f"debug log features: {self.features}")
            print(f"debug log labels: {self.labels}")
            print(f"debug log npy file: {npy_file_path}")

    def process_data(self):
        # get positive and negative samples
        positive_indexes = np.nonzero(self.labels)[0]
        negative_indexes = np.nonzero(self.labels == 0)[0]
        # upsampling positive samples
        num_positive_sample = len(positive_indexes)
        num_negative_sample = len(negative_indexes)
        print(f"num_positive_sample: {num_positive_sample}")
        print(f"total_num_samples: {num_positive_sample+num_negative_sample}")
        self.num_positive_samples += num_positive_sample
        self.num_negative_samples += num_negative_sample
        bool_con = False

        samples = []
        for i in range(self.num_samples):
            if num_positive_sample <= 0:
                print(f"warning! num_positive_sample < 0")
                bool_con = True
                break
            pivot_node_features = self.features[i:i+1, :]
            label = self.labels[i:i+1, :]
            children_features = self.get_children(i)
            edge_indexes = self.get_edge_indexes(children_features.shape[0])
            sample = (pivot_node_features, edge_indexes,
                      children_features, label)
            samples.append(sample)
            if self.debug_log:
                print(
                    f"debug log sample {i}, pivot_node_features: {pivot_node_features}")
                print(f"debug log sample {i}, label: {label}")
                print(
                    f"debug log sample {i}, children_features: {children_features}")
                print(f"debug log sample {i}, edge_indexes: {edge_indexes}")

        return samples, bool_con

    def get_children(self, i):
        children_indexes = self.divisors[i]
        children_features = []
        for index in children_indexes:
            if index >= 0:
                children_feature = self.features[index:index+1, :]
                children_features.append(children_feature)
            else:
                if self.fanin_nodes_type == 'zero':
                    children_feature = np.zeros(
                        (1, NUMFeatures), dtype=np.int32)
                elif self.fanin_nodes_type == 'one':
                    children_feature = np.ones(
                        (1, NUMFeatures), dtype=np.int32)
                elif self.fanin_nodes_type == 'remove':
                    continue
                children_features.append(children_feature)

        if len(children_features) == 0:
            children_features.append(np.ones((1, NUMFeatures), dtype=np.int32))
        return np.vstack(children_features)


if __name__ == '__main__':
    """
    test for graph data loader for single/multi domain binar class train
    """
    # data_loader = GraphDataLoader(
    #     './npy_data_mfs2/epfl/epfl_hard/test',
    #     'tmp_processed_data',
    #     processed_npy_path=None,
    #     sample_bool=False,
    #     train_type='train',
    #     sample_type='upsampling',
    #     load_type='default_order',
    #     depth=2,
    #     debug_log=False,
    #     fanin_nodes_type='remove'
    # )
    # for i in range(10):
    #     for batch in data_loader.graph_data_loader:
    #         # print(f"pivot node features: {batch.pivot_node_features.shape}")
    #         # print(f"edge_index features: {batch.edge_index.shape}")
    #         # print(f"children_node_features features: {batch.children_node_features.shape}")
    #         # print(f"label: {batch.label.shape}")
    #         # print(batch.edge_index)
    #         print(f"{i} th pivot node features: {batch.pivot_node_features}")
    #         # print(f"sample_x shape: {sample_x.shape}")
    #         # print(f"sample_y shape: {sample_y.shape}")
    #         break
    """
    test for graph data loader for resub operator
    """
    # data_loader = GraphResubDataLoader(
    #     './npy_data/epfl_arithmetic_control_mixed/mem_ctrl/test',
    #     'tmp_processed_data',
    #     processed_npy_path=None,
    #     train_type='train',
    #     max_batch_size=16,
    #     debug_log=False,
    #     fanin_nodes_type='remove'
    # )

    # for i in range(10):
    #     for batch in data_loader.graph_data_loader:
    #         # print(f"pivot node features: {batch.pivot_node_features.shape}")
    #         # print(f"edge_index features: {batch.edge_index.shape}")
    #         # print(
    #         #     f"children_node_features features: {batch.children_node_features.shape}")
    #         # print(f"label: {batch.label.shape}")
    #         print(f"{i} th pivot node features: {batch.pivot_node_features}")
    #         # print(batch.edge_index)
    #         break

    """
    test for graph data loader for multi class train
    """
    # from utils.utils import get_label_dict

    # label_dict = get_label_dict(
    #     './npy_data/iwls2005/train'
    # )
    # print(label_dict)
    # data_loader = GraphDataLoader(
    #     './npy_data/iwls2005/train',
    #     'tmp_processed_data',
    #     processed_npy_path=None,
    #     sample_bool=False,
    #     train_type='train',
    #     sample_type='upsampling',
    #     load_type='default_order',
    #     depth=1,
    #     debug_log=True,
    #     label_dict=label_dict
    # )


    """
    test for score data loader
    """
    data_loader = GraphDataLoader(
        './npy_data_mfs2/epfl/log2/test',
        'tmp_processed_data',
        processed_npy_path=None,
        train_type='test',
        max_batch_size=256,
        debug_log=False,
        fanin_nodes_type='remove'
    )
    for batch in data_loader.graph_data_loader:
        print(f"cur id: {batch.pivot_node_features[:,4]}")
    # data_loader = ScoreDataLoader(
    #     './npy_data_mfs2/epfl/log2/test',
    #     batch_size=256,
    #     operator_type='original',
    #     train_type='test'
    # )
    # for x, y in data_loader.test_data_loader:
    #     print(f"cur id: {x[:,4]}")

    # for i in range(10):
    #     for x, y in data_loader.batch_data_generator:
    #         print(f"cur {i}: {x}")
    #         break
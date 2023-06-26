# step1: call abc to collect data
# step2: read and preprocess csv
# step3: write save data to npy

import abc_py as abcPy
import numpy as np
import pandas as pd

import argparse
import os
import re

# logic synthesis flow optimization
# No.1: resyn2; resub -K 16 -N 3 -z -v; print_stats;
# No.2: resub -K 16 -N 3 -z -v; print_stats;


class DataCollector():
    def __init__(
        self,
        aig_data_path,
        save_dir,
        num_lo_flow,
        **resub_kwargs
    ):
        # init kwargs
        self.aig_data_path = aig_data_path
        self.save_dir = save_dir
        self.num_lo_flow = num_lo_flow
        self.resub_kwargs = resub_kwargs

        # init data
        self.end_to_end_stats = []
        self.features_list = []
        self.labels_list = []
        self.children = []
        self.parents = []
        self.traverse_id = []

    def run_abc_resub_seq(self):
        # multiple circuits run mfs2
        aig_files = os.listdir(self.aig_data_path)
        for aig_file in aig_files:
            print(f"current blif file: {aig_file} **************")
            self.run_resub_seq_aig_file(aig_file)

    def run_resub_seq_aig_file(self, aig_file):
        # single circuit repeat collect data
        aig_file_path = os.path.join(self.aig_data_path, aig_file)
        ############# default order collect data #############
        self._collect_data(aig_file_path)

        # ############# permutation 1 collect data #############
        # self._collect_data(1, blif_file_path)

        # ############# permutation 2 collect data #############
        # self._collect_data(2, blif_file_path)

        save_data = {
            "end_to_end_stats": self.end_to_end_stats,
            "features_list": self.features_list,
            "labels_list": self.labels_list,
            "children": self.children,
            "parents": self.parents
        }

        self.save_data(save_data, aig_file)
        self.reset_data()

    def _collect_data(self, aig_file_path):
        initStats, before_resub_stats, endStats = self._run_resub_seq_aig_file(
            aig_file_path)
        # process end-to-end stats
        new_stats_list = self._process_stats(
            [initStats, before_resub_stats, endStats])
        self.end_to_end_stats.append(new_stats_list)
        # process csv
        # features, labels, children, parents = self._process_csv(p)
        # self.features_list.append(features)
        # self.labels_list.append(labels)
        # self.children.append(children)
        # self.parents.append(parents)

    def _run_resub_seq_aig_file(self, aig_file_path):
        _abc = abcPy.AbcInterface()
        _abc.start()
        _abc.read(aig_file_path)
        initStats = _abc.getStats()
        if self.num_lo_flow == 1:
            _abc.resyn2()

        before_resub_stats = _abc.getStats()

        _abc.resub(**self.resub_kwargs)
        endStats = _abc.getStats()

        return initStats, before_resub_stats, endStats

    def save_data(self, save_data, blif_file):
        save_path = f'./npy_data/{self.save_dir}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_npy = f'{save_path}/save_data_total_{blif_file}_{self.save_dir}_flow_num_{self.num_lo_flow}.npy'
        np.save(save_npy, save_data)

    def reset_data(self):
        self.end_to_end_stats = []
        self.features_list = []
        self.labels_list = []
        self.children = []
        self.parents = []

    def get_key_value(self, stats, key_str):
        index = stats.index(key_str)
        if stats[index+1] == '=':
            num_and = stats[index+2]
        else:
            num_and = stats[index+1][1:]

        return float(num_and)

    def get_time_value(self, stats, key_str):
        index = stats.index(key_str)
        run_time = float(stats[index+1])
        return run_time

    def get_new_stats(self, stats, i):
        new_stats = {
            "_num_and": 0,
            "_num_lev": 0,
            "run_time": None
        }
        num_and = self.get_key_value(stats, 'and')
        num_lev = self.get_key_value(stats, 'lev')
        new_stats['_num_and'] = num_and
        new_stats['_num_lev'] = num_lev
        run_time = self.get_time_value(stats, 'elapse:')
        new_stats['run_time'] = run_time
        return new_stats

    def _process_stats(self, stats_list):
        # precess stats
        new_stats_list = []
        for i, stats in enumerate(stats_list):
            stats = stats.split()
            stats = [s for s in stats if s != '']
            print(f"the {i}-th stats: {stats}")
            new_stats = self.get_new_stats(stats, i)
            new_stats_list.append(new_stats)

        print(new_stats_list)
        return new_stats_list


class DataCollectorWithCSV(DataCollector):
    def __init__(
            self,
            aig_data_path,
            save_dir,
            num_lo_flow,
            csv_feature_path,
            csv_label_path,
            mode,
            normalize,
            **resub_kwargs
    ):
        DataCollector.__init__(
            self,
            aig_data_path,
            save_dir,
            num_lo_flow,
            **resub_kwargs
        )
        self.csv_feature_path = csv_feature_path
        self.csv_label_path = csv_label_path
        self.mode = mode
        self.normalize = normalize
        self.traverse_id = None
        self.divs_list = []

    def run_resub_seq_aig_file(self, aig_file):
        # single circuit repeat collect data
        aig_file_path = os.path.join(self.aig_data_path, aig_file)
        ############# default order collect data #############
        self._collect_data(aig_file_path)

        # ############# permutation 1 collect data #############
        # self._collect_data(1, blif_file_path)

        # ############# permutation 2 collect data #############
        # self._collect_data(2, blif_file_path)

        save_data = {
            "end_to_end_stats": self.end_to_end_stats,
            "features_list": self.features_list,
            "labels_list": self.labels_list,
            "divs_list": self.divs_list
        }

        self.save_data(save_data, aig_file)
        self.reset_data()

    def _collect_data(self, aig_file_path):
        if self.mode == 'online':
            initStats, before_resub_stats, endStats = self._run_resub_seq_aig_file(
                aig_file_path)
            # process end-to-end stats
            new_stats_list = self._process_stats(
                [initStats, before_resub_stats, endStats])
            self.end_to_end_stats.append(new_stats_list)
        # process csv
        features, labels, divs_ids = self._process_csv()
        self.features_list.append(features)
        self.labels_list.append(labels)
        self.divs_list.append(divs_ids)

    def _normalize(self, features):
        max_vals = np.max(features, axis=0)
        min_vals = np.min(features, axis=0)
        normalized_features = (features - min_vals) / \
            (max_vals - min_vals + 1e-3)

        print(f"debug log normalized features: {normalized_features}")
        return normalized_features

    def _process_csv(self):
        # process collected csv
        data_frame = pd.read_csv(self.csv_feature_path)
        data_frame_label = pd.read_csv(self.csv_label_path)
        print(f"debug log: {data_frame}")
        print(f"debug log: {data_frame_label}")
        numpy_data = data_frame.to_numpy()
        numpy_label_data = data_frame_label.to_numpy()
        # get features current node
        features = numpy_data[:, 0:8].astype(np.int32)
        self.traverse_id = features[0, 0]
        if self.normalize:
            features = self._normalize(features)
        # get children features
        child0features = self.get_child_feature(features, numpy_data, 0)
        child1features = self.get_child_feature(features, numpy_data, 1)
        # concat features
        features = np.concatenate(
            (features, child0features, child1features),
            axis=1
        )

        # get labels
        labels = self._get_labels(numpy_data, numpy_label_data)

        # get divs ids
        divs_ids = self.get_div_ids(numpy_data)
        return features, labels, divs_ids

    def get_child_feature(self, features, numpy_data, i):
        # child_features = np.empty(
        #     (features.shape[0], features.shape[1]-2), dtype=np.int32)
        child_features = np.empty(
            (features.shape[0], features.shape[1]-2))
        # child_features = np.empty(
        #     (features.shape[0], features.shape[1]))
        first_id = self.traverse_id
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

    def _get_labels(self, numpy_data_feature, numpy_label_data):
        # implemented based on numpy faster
        # generated labels
        labels = np.zeros((numpy_data_feature.shape[0], 1), dtype=np.int32)

        # features
        features_traverse_id = numpy_data_feature[:, 0:1].astype(np.int32)
        print(f"debug log features shape: {features_traverse_id.shape}")
        # collected labels
        labels_traverse_id = numpy_label_data[:, 0:1].astype(np.int32)
        collected_labels = numpy_label_data[:, 1:2].astype(np.int32)
        print(f"debug log collected_labels shape: {collected_labels.shape}")
        # get positive traverse id
        label_nonzero = np.nonzero(collected_labels)
        print(f"debug log label_nonzero: {label_nonzero}")
        positive_traverse_id = labels_traverse_id[label_nonzero]
        print(f"debug log positive_traverse_id: {positive_traverse_id}")
        print(f"debug log positive_traverse_id: {positive_traverse_id.shape}")
        postive_indexes = [np.where(features_traverse_id == traverse_id)[
            0] for traverse_id in positive_traverse_id]
        if postive_indexes:
            postive_indexes = np.vstack(postive_indexes)
            print(f"debug log postive_indexes: {postive_indexes}")
            print(f"debug log postive_indexes shape: {postive_indexes.shape}")
            labels[postive_indexes] = 1
        labels = labels.astype(np.int32)

        return labels

    def get_div_ids(self, numpy_data):
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

    def reset_data(self):
        self.end_to_end_stats = []
        self.features_list = []
        self.labels_list = []
        self.divs_list = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test scripts')
    parser.add_argument('--aig_data_path', type=str,
                        default='../../benchmarks/arithmetic/test_blif')
    parser.add_argument('--k', type=int, default=16)  # max window size
    parser.add_argument('--n', type=int, default=3)  # max TFO cone num
    parser.add_argument('--z', action='store_false')
    parser.add_argument('--save_dir', type=str, default='arith_resub')
    parser.add_argument('--num_lo_flow', type=int, default=1)
    parser.add_argument('--csv_feature_path', type=str,
                        default='features_resub.csv')
    parser.add_argument('--csv_label_path', type=str,
                        default='labels_resub.csv')
    parser.add_argument('--mode', type=str,
                        default='online')
    parser.add_argument('--normalize', action='store_true')

    args = parser.parse_args()
    resub_kwargs = {
        "k": args.k,
        "n": args.n,
        "z": args.z,
        "v": False
    }

    data_collector = DataCollectorWithCSV(
        args.aig_data_path,
        args.save_dir,
        args.num_lo_flow,
        args.csv_feature_path,
        args.csv_label_path,
        args.mode,
        args.normalize,
        **resub_kwargs
    )

    data_collector.run_abc_resub_seq()

# step1: call abc to collect data
# step2: read and preprocess csv
# step3: write save data to npy

import abc_py as abcPy
import numpy as np
import torch
import pandas as pd

import argparse
import os
import re

# logic synthesis flow optimization
# No.1: strash; dch; if -K 4; mfs2
# No.2: strash; if -K 4; mfs2
# No.3: strash; compress2rs; if -K 4; mfs2
# No.4: strash; compress2rs; if -K 4; lutpack; mfs2
# No.5: strash; rewrite; if; mfs2
# No.6: strash; balance; if; mfs2
# No.7: strash; refactor; if; mfs2
# compress2rs

# strash; resyn2; resyn2; if -K 4; mfs2
# strash; resyn2; resyn2; if -K 4; mfs2; mfs2


class DataCollector():
    def __init__(
        self,
        blif_data_path,
        save_dir,
        repeat_times,
        csv_feature_path,
        csv_label_path,
        csv_truth_table_path,
        dch,
        num_lo_flow,
        **mfs2_kwargs
    ):
        # init kwargs
        self.blif_data_path = blif_data_path
        self.save_dir = save_dir
        self.repeat_times = repeat_times
        self.csv_feature_path = csv_feature_path
        self.csv_label_path = csv_label_path
        self.csv_truth_table_path = csv_truth_table_path
        self.dch = dch
        self.num_lo_flow = num_lo_flow
        print(f"dch: {dch}")
        self.mfs2_kwargs = mfs2_kwargs

        # init data
        self.end_to_end_stats = []
        self.features_list = []
        self.labels_list = []
        self.children = []
        self.parents = []
        self.traverse_id = []

    def run_abc_mfs2_seq(self):
        # multiple circuits run mfs2
        blif_files = os.listdir(self.blif_data_path)
        for blif_file in blif_files:
            print(f"current blif file: {blif_file} **************")
            self.run_mfs2_seq_blif_file(blif_file)

    def _collect_data(self, p, blif_file_path):
        self.mfs2_kwargs['p'] = p
        if p == 0:
            repeat_times = 1
        else:
            repeat_times = self.repeat_times

        for _ in range(repeat_times):
            initStats, before_mfs2_stats, endStats = self._run_mfs2_seq_blif_file(
                blif_file_path)
            # process end-to-end stats
            new_stats_list = self._process_stats(
                [initStats, before_mfs2_stats, endStats])
            self.end_to_end_stats.append(new_stats_list)
            # process csv
            features, labels, children, parents = self._process_csv(p)
            self.features_list.append(features)
            self.labels_list.append(labels)
            self.children.append(children)
            self.parents.append(parents)

    def save_data(self, save_data, blif_file):
        save_path = f'./npy_data/{self.save_dir}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_npy = f'{save_path}/save_data_total_{blif_file}_{self.save_dir}_flow_num_{self.num_lo_flow}.npy'
        np.save(save_npy, save_data)

    def run_mfs2_seq_blif_file(self, blif_file):
        # single circuit repeat collect data
        blif_file_path = os.path.join(self.blif_data_path, blif_file)
        ############# default order collect data #############
        self._collect_data(0, blif_file_path)

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

        self.save_data(save_data, blif_file)
        self.reset_data()

    def reset_data(self):
        self.end_to_end_stats = []
        self.features_list = []
        self.labels_list = []
        self.children = []
        self.parents = []

    def _run_mfs2_seq_blif_file(self, blif_file_path):
        _abc = abcPy.AbcInterface()
        _abc.start()
        _abc.read(blif_file_path)
        initStats = _abc.getStats()
        if self.num_lo_flow == 1:
            # strash; dch; If -C 12; mfs2;
            if self.dch:
                _abc.dch(v=False)
        elif self.num_lo_flow == 2:
            # strash; If -C 12; mfs2;
            pass
        elif self.num_lo_flow in [3, 4]:
            # strash; compress2rs; If -C 12; mfs2;
            # strash; compress2rs; If -C 12; lutpck; mfs2;
            _abc.compress2rs()
        elif self.num_lo_flow == 5:
            _abc.rewrite()
        elif self.num_lo_flow == 6:
            _abc.balance()
        elif self.num_lo_flow == 7:
            _abc.refactor()
        _abc.If(C=12, v=False)
        before_mfs2_stats = _abc.getStats()
        if self.num_lo_flow == 4:
            _abc.lutpack(v=False)
        _abc.mfs2(**self.mfs2_kwargs)
        endStats = _abc.getStats()

        return initStats, before_mfs2_stats, endStats

    def _process_stats(self, stats_list):
        # precess stats
        new_stats_list = []
        for i, stats in enumerate(stats_list):
            new_stats = {
                "_num_and": 0,
                "_num_lev": 0,
                "_num_edge": 0,
                "run_time": None
            }
            stats = re.sub(r'\x1b\[\d(;\d+)?m', '', stats)
            stats = stats.split()
            stats = [s for s in stats if s != '']
            # and
            if 'and' in stats:
                index = stats.index('and')
            elif 'nd' in stats:
                index = stats.index('nd')
            if stats[index+1] != '=':
                new_stats["_num_and"] = int(stats[index+1][1:])
            else:
                new_stats["_num_and"] = int(stats[index+2])
            # lev
            index = stats.index('lev')
            if stats[index+1] != '=':
                new_stats["_num_lev"] = int(stats[index+1][1:])
            else:
                new_stats["_num_lev"] = int(stats[index+2])
            # edge
            if 'edge' in stats:
                index = stats.index('edge')
                if stats[index+1] != '=':
                    new_stats["_num_edge"] = int(stats[index+1][1:])
                else:
                    new_stats["_num_edge"] = int(stats[index+2])
            # time
            if 'elapse:' in stats:
                index = stats.index('elapse:')
                new_stats["run_time"] = float(stats[index+1])
            new_stats_list.append(new_stats)
            print(f"the {i}-th stats: {new_stats}")
        return new_stats_list

    def _process_stats_backup(self, stats_list):
        # precess stats
        new_stats_list = []
        for i, stats in enumerate(stats_list):
            new_stats = {
                # "_num_in": 0,
                # "_num_out": 0,
                "_num_and": 0,
                "_num_lev": 0,
                "_num_edge": 0,
                "_num_nodes_changed": 0,
                "run_time": None
            }
            stats = re.sub(r'\x1b\[\d(;\d+)?m', '', stats)
            stats = stats.split()
            stats = [s for s in stats if s != '']
            print(f"{i}th stats *************")
            print(stats)
            # index = stats.index('i/o')
            # new_stats["_num_in"] = int(stats[index+2][:-1])
            # new_stats["_num_out"] = int(stats[index+3])
            # and
            if 'and' in stats:
                index = stats.index('and')
            elif 'nd' in stats:
                index = stats.index('nd')
            if stats[index+1] != '=':
                new_stats["_num_and"] = int(stats[index+1][1:])
            else:
                new_stats["_num_and"] = int(stats[index+2])
            # lev
            index = stats.index('lev')
            if stats[index+1] != '=':
                new_stats["_num_lev"] = int(stats[index+1][1:])
            else:
                new_stats["_num_lev"] = int(stats[index+2])
            # edge
            if 'edge' in stats:
                index = stats.index('edge')
                if stats[index+1] != '=':
                    new_stats["_num_edge"] = int(stats[index+1][1:])
                else:
                    new_stats["_num_edge"] = int(stats[index+2])
            # nodes changed
            if 'has' in stats:
                index = stats.index('has')
                new_stats["_num_nodes_changed"] = int(stats[index+1])
            # all time
            if 'ALL' in stats:
                index = stats.index('ALL')
                new_stats["run_time"] = float(stats[index+2])
            # print(f"{i}th new stats *************")
            # print(new_stats)
            new_stats_list.append(new_stats)
        return new_stats_list

    def _get_labels(self, numpy_data_feature, numpy_label_data):
        # labels = np.zeros((numpy_data_feature.shape[0],1), dtype=np.int32)
        # features_traverse_id = list(numpy_data_feature[:,5].astype(np.int32))
        # labels_traverse_id = list(numpy_label_data[:,0].astype(np.int32))
        # collected_labels = numpy_label_data[:,-2:-1].astype(np.int32)
        # for i in range(len(features_traverse_id)):
        #     try:
        #         index = labels_traverse_id.index(features_traverse_id[i])
        #         labels[i] = collected_labels[index]
        #     except:
        #         # print(f"debug log: {i}th id not in collected labels")
        #         pass

        # return labels

        # implemented based on numpy faster
        # generated labels
        labels = np.zeros((numpy_data_feature.shape[0], 1), dtype=np.int32)

        # features
        features_traverse_id = numpy_data_feature[:, 5:6].astype(np.int32)
        print(f"features shape: {features_traverse_id.shape}")
        # collected labels
        labels_traverse_id = numpy_label_data[:, 0:1].astype(np.int32)
        collected_labels = numpy_label_data[:, -2:-1].astype(np.int32)
        print(f"collected_labels shape: {collected_labels.shape}")
        # get positive traverse id
        label_nonzero = np.nonzero(collected_labels)
        print(f"label_nonzero: {label_nonzero}")
        positive_traverse_id = labels_traverse_id[label_nonzero]
        print(f"positive_traverse_id: {positive_traverse_id}")
        print(f"positive_traverse_id: {positive_traverse_id.shape}")
        postive_indexes = [np.where(features_traverse_id == traverse_id)[
            0] for traverse_id in positive_traverse_id]
        postive_indexes = np.vstack(postive_indexes)
        print(f"postive_indexes: {postive_indexes}")
        print(f"postive_indexes shape: {postive_indexes.shape}")
        labels[postive_indexes] = 1
        labels = labels.astype(np.int32)

        return labels

    def _process_csv(self, p):
        # process collected csv
        data_frame = pd.read_csv(self.csv_feature_path)
        data_frame_label = pd.read_csv(self.csv_label_path)
        data_frame_truth_table = pd.read_csv(self.csv_truth_table_path)
        # print(data_frame)
        # print(data_frame_label)
        # print(data_frame_truth_table)
        numpy_data = data_frame.to_numpy()
        numpy_label_data = data_frame_label.to_numpy()
        numpy_data_truth_table = data_frame_truth_table.to_numpy()
        # get features current node
        features = numpy_data[:, 1:6].astype(np.int32)
        labels = self._get_labels(numpy_data, numpy_label_data)
        # get features truth table
        features_truth_table = numpy_data_truth_table[:, 1:]
        # get features x and label y
        features = np.concatenate((features, features_truth_table), axis=1)
        # print(f"features shape: {features.shape}")
        children = numpy_data[:, -2]
        parents = numpy_data[:, -1]
        if p == 0:
            self.traverse_id = list(numpy_data[:, 0:1])
        return features, labels, children, parents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test scripts')
    parser.add_argument('--blif_data_path', type=str,
                        default='../../benchmarks/arithmetic/test_blif')
    parser.add_argument('--M', type=int, default=5000)  # max window size
    parser.add_argument('--W', type=int, default=4)  # max TFO cone num
    parser.add_argument('--p', type=int, default=0)  # max TFO cone num
    parser.add_argument('--l', action='store_true')
    parser.add_argument('--dch', action='store_false')
    parser.add_argument('--save_dir', type=str, default='arith')
    parser.add_argument('--i', type=int, default=0)  # max TFO cone num
    parser.add_argument('--repeat_times', type=int, default=1)
    parser.add_argument('--csv_feature_path', type=str,
                        default='./features.csv')
    parser.add_argument('--csv_label_path', type=str, default='./labels.csv')
    parser.add_argument('--csv_truth_table_path', type=str,
                        default='./truth_table.csv')
    parser.add_argument('--num_lo_flow', type=int, default=1)

    args = parser.parse_args()
    mfs2_kwargs = {
        "W": args.W,
        "M": args.M,
        "p": args.p,
        "l": args.l,
        "v": False,
        "w": False
    }

    data_collector = DataCollector(
        args.blif_data_path,
        args.save_dir,
        args.repeat_times,
        args.csv_feature_path,
        args.csv_label_path,
        args.csv_truth_table_path,
        args.dch,
        args.num_lo_flow,
        **mfs2_kwargs
    )

    data_collector.run_abc_mfs2_seq()

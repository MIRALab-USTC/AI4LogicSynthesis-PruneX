import argparse
import json
import os
import subprocess
import numpy as np
import re

def process_output(output):
    stats = {
        'and': [],
        'edge': [],
        'lev': [],
        'run_time': [],
        'all_run_time': [],
        'time_process_csv': [],
        'time_process_children': [],
        'time_process_graph_samples': [],
        'time_load_graph_model': [],
        'time_policy_inference': []
    }
    output = output.decode()
    output = re.sub(r'\x1b\[\d(;\d+)?m', '', output)
    output = output.split('\n')
    output = output[2:]
    print(f"output: {output}")
    for o_str in output:
        if 'nd' in o_str and 'seconds' not in o_str:
            o_str_list = o_str.split()
            print(f"o str list: {o_str_list}")
            and_index = o_str_list.index('nd')
            if o_str_list[and_index+1] != '=':
                stats['and'].append(int(o_str_list[and_index+1][1:]))
            else:
                stats['and'].append(int(o_str_list[and_index+2]))
        if 'lev' in o_str:
            o_str_list = o_str.split()
            lev_index = o_str_list.index('lev')
            if o_str_list[lev_index+1] != '=':
                stats['lev'].append(int(o_str_list[lev_index+1][1:]))
            else:
                stats['lev'].append(int(o_str_list[lev_index+2]))
        if 'edge' in o_str:
            o_str_list = o_str.split()
            lev_index = o_str_list.index('edge')
            if o_str_list[lev_index+1] != '=':
                stats['edge'].append(int(o_str_list[lev_index+1][1:]))
            else:
                stats['edge'].append(int(o_str_list[lev_index+2]))
        if 'elapse' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('elapse:')
            run_time = float(o_str_list[time_index+1])
            stats['run_time'].append(run_time)
        if 'total' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('total:')
            all_run_time = float(o_str_list[time_index+1])
            stats['all_run_time'].append(all_run_time)
        if 'process_csv' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_csv:')
            time_process_csv = float(o_str_list[time_index+1])
            stats['time_process_csv'].append(time_process_csv)
        if 'process_children' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_children:')
            time_process_children = float(o_str_list[time_index+1])
            stats['time_process_children'].append(time_process_children)
        if 'process_graph_samples' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('process_graph_samples:')
            time_process_graph_samples = float(o_str_list[time_index+1])
            stats['time_process_graph_samples'].append(time_process_graph_samples)
        if 'load_graph_model' in o_str:
            o_str_list = o_str.split()
            time_index = o_str_list.index('load_graph_model:')
            time_load_graph_model = float(o_str_list[time_index+1])
            stats['time_load_graph_model'].append(time_load_graph_model)
        if 'inference:' in o_str:
            o_str_list = o_str.split()
            print(f"o str list: {o_str_list}")
            time_index = o_str_list.index('inference:')
            time_inference = float(o_str_list[time_index+1])
            stats['time_policy_inference'].append(time_inference)
    return stats

def save_data(save_data, save_dir, test_blif_name):
    save_path = f'./npy_data/mfs2_time/{save_dir}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_npy = f'{save_path}/stats_{test_blif_name}.npy'
    np.save(save_npy, save_data)


def get_command(test_blif_path, dch, num_lo_flow):
    # flow1: 1 times mfs2
    # flow2: 2 times mfs2
    # flow3: 1 times mfs2 large window size
    # flow4: mfs2; lutpack; mfs2 optimization
    # flow5: mfs2; strash; if -C 12; mfs2 optimization
    # flow6: mfs2; lutpack optimization
    if num_lo_flow == 1:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 2:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 3:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 50000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 50000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 4:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; lutpack; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; lutpack; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 5:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; strash; if -C 12; mfs2 -W 4 -M 5000 -l -p 0; print_stats -t'"
    elif num_lo_flow == 6:
        if dch:
            command = f"./abc -c 'r {test_blif_path}; strash; dch; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; lutpack; print_stats -t'"
        else:
            command = f"./abc -c 'r {test_blif_path}; strash; if -C 12; print_stats -t; mfs2 -W 4 -M 5000 -l -p 0; lutpack; print_stats -t'"

    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test scripts')
    parser.add_argument('-sel_percents', nargs="+", type=float)

    parser.add_argument('--model_path', type=str,
                        default="./models/itr_2800.pkl")
    parser.add_argument('--test_blif_path', type=str,
                        default="/datasets/ai4eda/benchmarks/arithmetic/test_blif/log2.blif")
    parser.add_argument('--test_blif', type=str,
                        default="log2")
    parser.add_argument('--dch', action='store_false')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--num_lo_flow', type=int,
                        default=1)
    parser.add_argument('--save_dir', type=str, default='log2_online_test')

    args = parser.parse_args()

    # GLOBAL VAR
    SEL_PERCENTS = args.sel_percents
    DEVICE = "cpu"

    # command
    command = get_command(args.test_blif_path, args.dch, args.num_lo_flow)

    stats_dict = {}
    for sel_percent in SEL_PERCENTS:
        json_kwargs = {
            "GCNMODEL": args.model_path,
            "DEVICE": DEVICE,
            "SEL_PERCENT": sel_percent,
            "RANDOM": args.random,
            "NORMALIZE": args.normalize
        }
        with open('./configs/configs.json', 'w') as f:
            json.dump(json_kwargs, f)
        output = subprocess.check_output(command, shell=True)
        stats = process_output(output)
        k = f"sel_percent_{json_kwargs['SEL_PERCENT']}"
        stats_dict[k] = stats
    print(f"cur blif: {args.test_blif_path}, {stats_dict}")
    save_data(stats_dict, args.save_dir, args.test_blif)

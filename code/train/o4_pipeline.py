import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import o2_data_loader as data_loader
import o3_trainer as trainer
import utils.score_policy as score_policy
import utils.gcn_policy as gcn_policy

from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_global_seed, get_label_dict

DEVICE = 'cuda:0'


@hydra.main(config_path="configs")
def pipeline(cfg: DictConfig) -> None:
    # set seed
    print(os.getcwd())
    seed = set_global_seed(cfg.config_groups.exp_kwargs.seed)
    # set tensorboard logger
    tb_logger = SummaryWriter(cfg.config_groups.exp_kwargs.base_log_dir)
    # init policy
    if cfg.config_groups.policy.policy_type == "gcn":
        policy = getattr(gcn_policy, cfg.config_groups.policy['class'])(
            **cfg.config_groups.policy.kwargs).to(DEVICE)
    elif cfg.config_groups.policy.policy_type == "score":
        policy = getattr(score_policy, cfg.config_groups.policy['class'])(
            **cfg.config_groups.policy.kwargs).to(DEVICE)
    else:
        raise NotImplementedError
    # load attention policy
    if cfg.config_groups.exp_kwargs.attention:
        attention_policy = getattr(gcn_policy, cfg.config_groups.attention_policy['class'])(
            **cfg.config_groups.attention_policy.kwargs).to(DEVICE)
        attention_policy.load_state_dict(torch.load(
            cfg.config_groups.attention_policy.model_f))
        attention_policy.eval()
    else:
        attention_policy = None

    # init data loader
    if cfg.config_groups.exp_kwargs.multi_class:
        label_dict_train = get_label_dict(
            cfg.config_groups.train_data_loader.kwargs.npy_data_path
        )
        label_dict_test = get_label_dict(
            cfg.config_groups.test_data_loader.kwargs.npy_data_path
        )
        cfg.config_groups.train_data_loader.kwargs.label_dict = label_dict_train
        cfg.config_groups.test_data_loader.kwargs.label_dict = label_dict_test
    # train data loader
    if cfg.config_groups.exp_kwargs.domain == 'multi':
        train_data_loader = []
        npy_data_path = cfg.config_groups.train_data_loader.kwargs.npy_data_path
        for npy_file in os.listdir(npy_data_path):
            print(npy_file)
            cur_npy_file_path = os.path.join(npy_data_path, npy_file)
            cfg.config_groups.train_data_loader.kwargs.npy_data_path = cur_npy_file_path
            train_data_loader.append(
                getattr(data_loader, cfg.config_groups.train_data_loader['class'])(
                    **cfg.config_groups.train_data_loader.kwargs)
            )
    elif cfg.config_groups.exp_kwargs.domain == 'single':
        train_data_loader = getattr(data_loader, cfg.config_groups.train_data_loader['class'])(
            **cfg.config_groups.train_data_loader.kwargs)
    # test data loader
    if cfg.config_groups.test_data_loader.kwargs.npy_data_path.endswith('.npy'):
        test_data_loader = [(getattr(data_loader, cfg.config_groups.test_data_loader['class'])(
            **cfg.config_groups.test_data_loader.kwargs), cfg.config_groups.test_data_loader.kwargs.npy_data_path[34:])]
    else:
        test_data_loader = []
        npy_data_path = cfg.config_groups.test_data_loader.kwargs.npy_data_path
        for npy_file in os.listdir(npy_data_path):
            print(npy_file)
            cur_npy_file_path = os.path.join(npy_data_path, npy_file)
            cfg.config_groups.test_data_loader.kwargs.npy_data_path = cur_npy_file_path
            test_data_loader.append(
                (getattr(data_loader, cfg.config_groups.test_data_loader['class'])(
                    **cfg.config_groups.test_data_loader.kwargs), npy_file)
            )
    # init trainer
    train_agent = getattr(trainer, cfg.config_groups.trainer['class'])(
        train_data_loader,
        test_data_loader,
        policy,
        tb_logger,
        attention_model=attention_policy,
        **cfg.config_groups.trainer.kwargs
    )
    train_agent.train()


if __name__ == '__main__':
    pipeline()

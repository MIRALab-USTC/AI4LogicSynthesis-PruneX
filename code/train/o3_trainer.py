import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class Trainer():
    def __init__(
        self,
        data_loader,
        evaluate_data_loader,
        policy,
        tb_logger,
        # attention module
        attention_model=None,
        loss_type='focal',
        focal_gamma=2,
        policy_type='score',
        evaluate_freq=1,
        epochs=100,
        lr=1e-4,
        batch_size=128,
        optim_class='Adam',
        device='cuda:0',
        model_save_dir='./models',
        # lr decay
        use_lr_decay=True,
        step_size=20,
        gamma=0.92,
        # weight decay
        weight_decay=1e-5,
        # multi domain loss and inference
        infer_mean_max='max',
        multi_loss='mean',
        drop_num=1,
        # ensemble gcn
        single_domain_ensemble=False,
        # debug log
        debug_log=True,
        # multi class
        label_smoothing=0.2
    ):
        # TODO:
        # 1. add lr decay
        # 2. add early stop
        # 3. add regularization
        # 4. add train test split

        self.data_loader = data_loader
        self.evaluate_data_loader = evaluate_data_loader
        self.policy = policy
        self.policy_type = policy_type
        self.evaluate_freq = evaluate_freq
        self.infer_mean_max = infer_mean_max
        self.multi_loss = multi_loss
        self.drop_num = drop_num
        self.single_domain_ensemble = single_domain_ensemble
        self.debug_log = debug_log
        self.model_save_dir = model_save_dir
        self.attention_model = attention_model
        self.weight_decay = weight_decay

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        # optimizer
        self.optim_class = optim_class
        if isinstance(optim_class, str):
            optim_class = eval('optim.'+optim_class)
            self.optim_class = optim_class

        self.policy_optimizer = optim_class(
            self.policy.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # lr decayer
        self.use_lr_decay = use_lr_decay
        self.step_size = step_size
        self.gamma = gamma
        if self.use_lr_decay:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.policy_optimizer,
                step_size=self.step_size,
                gamma=self.gamma
            )
        # self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        if self.loss_type == 'focal':
            self.loss = self.focal_loss
        elif self.loss_type == 'binary_classification':
            self.loss = torch.nn.BCELoss(reduction='mean')
        elif self.loss_type == 'multi_classification':
            self.loss = torch.nn.CrossEntropyLoss(
                reduction='mean', label_smoothing=label_smoothing)
        # tensorboard
        self.tb_logger = tb_logger

        self.step = 0

    def focal_loss(self, predict_y, target_y):
        if self.single_domain_ensemble:
            target_y = target_y.repeat(1, predict_y.shape[1])
        epilson = 1e-3
        alpha_positive = (self.data_loader.num_negative_samples+1e-3) / (
            self.data_loader.num_positive_samples+self.data_loader.num_negative_samples)
        alpha_negative = (self.data_loader.num_positive_samples+1e-3) / (
            self.data_loader.num_positive_samples+self.data_loader.num_negative_samples)
        # positive
        weight_positive = torch.pow(1.-predict_y+epilson, self.focal_gamma)
        focal_positive = -alpha_positive * \
            weight_positive * torch.log(predict_y+epilson)
        loss_positive = target_y * focal_positive
        if self.debug_log:
            print(
                f"debug log focal loss weight positive: {weight_positive.shape}")
            print(
                f"debug log focal loss focal_positive: {focal_positive.shape}")
            print(f"debug log focal loss loss_positive: {loss_positive.shape}")
        # negative
        weight_negative = torch.pow(predict_y, self.focal_gamma)
        focal_negative = -alpha_negative * \
            weight_negative * torch.log(1.-predict_y+epilson)
        loss_negative = (1.-target_y) * focal_negative

        loss = torch.mean(loss_positive+loss_negative)
        print(f"debug log focal loss: {loss.shape}")
        return loss

    def train_epoch(self, epoch):
        loss_all = 0
        cnt = 0
        if self.policy_type == 'score':
            for sample_x, sample_y in self.data_loader.batch_data_generator:
                sample_x = sample_x.float().to(self.device)
                sample_y = sample_y.float().to(self.device)
                predict_y = self.policy(sample_x)
                print(f"predicy_y shape: {predict_y.shape}")
                print(f"predicy_y: {predict_y}")
                print(f"sample_y shape: {sample_y.shape}")
                print(f"sample_y: {sample_y}")

                batch_loss = self.loss(predict_y, sample_y)
                print(batch_loss)
                print(f"epoch {epoch}, step {cnt+1}, loss {batch_loss.item()}")
                break
            # update
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()

            # lr decay
            if self.use_lr_decay:
                self.lr_scheduler.step()

            loss_all += batch_loss
            cnt += 1
            if self.step % self.evaluate_freq == 0:
                self.evaluate(self.step, batch_loss,
                              evaluate_type='test_datasets')
                self.save_checkpoint(self.step)
            self.step += 1
        elif self.policy_type == 'gcn':
            for batch in self.data_loader.graph_data_loader:
                batch_pivot_node_features = batch.pivot_node_features.to(
                    self.device)
                batch_children_node_features = batch.children_node_features.to(
                    self.device)
                batch_edge_indexes = batch.edge_index.to(self.device)
                labels = batch.label.float().to(self.device)
                predict_y = self.policy(
                    batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
                batch_loss = self.loss(predict_y, labels)
                print(f"epoch {epoch}, step {cnt+1}, loss {batch_loss.item()}")
                break
            # update
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()

            # lr decay
            if self.use_lr_decay:
                self.lr_scheduler.step()
            
            loss_all += batch_loss
            cnt += 1
            if self.step % self.evaluate_freq == 0:
                if self.evaluate_data_loader is None:
                    # evaluate on training datasets, only for graph datasets
                    self.evaluate(self.step, batch_loss)
                else:
                    # evaluate on testing datasets, only for graph datasets
                    self.evaluate_graph(self.step, batch_loss)
                self.save_checkpoint(self.step)
            self.step += 1
        return loss_all / cnt

    def get_accuracy(self, predict_scores, default_order_label, num_total_samples):
        cnt = 0
        for index in range(default_order_label.shape[0]):
            if predict_scores[index] > 0.5 and default_order_label[index] == 1:
                cnt += 1
            if predict_scores[index] <= 0.5 and default_order_label[index] == 0:
                cnt += 1

        return cnt / num_total_samples

    def get_confusion_matrix(self, predict_scores, default_order_label):
        positive_indexes = np.nonzero(default_order_label)[0]
        negative_indexes = np.nonzero(default_order_label == 0)[0]

        predict_scores = predict_scores - 0.5
        predict_scores = np.maximum(predict_scores, 0)
        predict_positive_indexes = np.nonzero(predict_scores)[0]
        predict_negative_indexes = np.nonzero(predict_scores == 0)[0]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        # print(positive_indexes)
        print(f"predict positive indexes: {predict_positive_indexes}")
        print(f"label positive indexes: {positive_indexes}")
        num_positive_indexes = len(predict_positive_indexes)
        tp_indexes = list(set(predict_positive_indexes).intersection(positive_indexes))
        tp = len(tp_indexes)
        fp = len(predict_positive_indexes) - tp 

        tn_indexes = list(set(predict_negative_indexes).intersection(negative_indexes))
        tn = len(tn_indexes)
        fn = len(predict_negative_indexes) - tn
        # for ind in predict_positive_indexes:
        #     if ind in positive_indexes:
        #         tp += 1
        #     elif ind in negative_indexes:
        #         fp += 1

        # for ind in predict_negative_indexes:
        #     if ind in negative_indexes:
        #         tn += 1
        #     elif ind in positive_indexes:
        #         fn += 1

        return tp, tn, fp, fn, num_positive_indexes

    def get_positive_scores(self, scores):
        tmp_scores = scores - 0.5
        tmp_scores = np.maximum(tmp_scores, 0)
        positive_indexes = np.nonzero(tmp_scores)[0]

        return scores[positive_indexes]

    def evaluate(self, epoch, loss, evaluate_type='train_datasets'):
        # log
        self.tb_logger.add_scalar('train loss', loss.item(), global_step=epoch)
        # true label

        if evaluate_type == 'train_datasets':
            test_data_loader = self.data_loader
        elif evaluate_type == 'test_datasets':
            test_data_loader = self.evaluate_data_loader

        for evaluate_data_loader, npy_file in test_data_loader:
            predict_scores_list = []
            default_order_label = []
            for sample_x, sample_y in evaluate_data_loader.test_data_loader:
                sample_x = sample_x.float().to(self.device)
                # sample_y = sample_y.float().to(self.device)
            # default_order_features = evaluate_data_loader.dataset['default_order_features']
            # default_order_label = evaluate_data_loader.dataset['default_order_label']

            # predict label
            # default_order_features = torch.from_numpy(
            #     default_order_features).float().to(self.device)
                with torch.no_grad():
                    predict_scores = self.policy(sample_x)
                predict_scores = predict_scores.cpu().detach().numpy()
                predict_scores_list.append(predict_scores)
                default_order_label.append(sample_y.numpy())

            predict_scores = np.vstack(predict_scores_list)
            default_order_label = np.vstack(default_order_label)
            predict_scores = np.mean(predict_scores, axis=1, keepdims=True)
            ascending_indexes = np.argsort(
                predict_scores.squeeze())  # default ascending order

            # default_order_label = evaluate_data_loader.dataset['default_order_label']
            print(f"predict_scores: {predict_scores}")
            self.tb_logger.add_histogram(
                'predict scores', predict_scores, global_step=epoch)
            # accuracy
            num_total_samples = default_order_label.shape[0]
            # accuracy = self.get_accuracy(
            #     predict_scores, default_order_label, num_total_samples)
            # precision and recall
            print(f"before confusion matrix scores: {predict_scores}")
            tp, tn, fp, fn, num_predict_positive_indexes = self.get_confusion_matrix(
                predict_scores, default_order_label)
            precision = tp / (tp + fp + 1e-3)
            recall = tp / (tp + fn + 1e-3)

            # if self.get_positive_scores(predict_scores):
            #     self.tb_logger.add_histogram('predict scores positive', self.get_positive_scores(predict_scores), global_step=epoch)
            # top k accuracy
            positive_indexes = np.nonzero(default_order_label)[0]
            num_positive_indexes = positive_indexes.shape[0]
            # with torch.no_grad():
            #     ascending_indexes, predict_scores = self.policy.inference(
            #         default_order_features)
            total_num = num_total_samples
            cnt_top_k = [0 for _ in range(9)]
            for i in range(9):
                percent = 0.1 * (i+1)
                sel_num = int(total_num * percent)
                sel_indexes = ascending_indexes[total_num-sel_num:]
                cnt = 0
                inter_indexes = list(set(sel_indexes).intersection(positive_indexes))
                cnt = len(inter_indexes)
                # for index in positive_indexes:
                #     if index in sel_indexes:
                #         cnt += 1
                cnt_top_k[i] = cnt

            # self.tb_logger.add_scalar(
            #     f'accuracy {npy_file}', accuracy, global_step=epoch)
            self.tb_logger.add_scalar(
                f'precision {npy_file}', precision, global_step=epoch)
            self.tb_logger.add_scalar(
                f'recall {npy_file}', recall, global_step=epoch)
            self.tb_logger.add_scalar(
                f'num predict postive {npy_file}', num_predict_positive_indexes / num_total_samples, global_step=epoch)
            for i in range(9):
                self.tb_logger.add_scalar(
                    f'top {i+1}0% accuracy {npy_file}', cnt_top_k[i] / num_positive_indexes, global_step=epoch)

    def evaluate_graph(self, epoch, loss):
        # log
        self.tb_logger.add_scalar('train loss', loss.item(), global_step=epoch)
        # true label
        for evaluate_data_loader, npy_file in self.evaluate_data_loader:
            predict_scores_list = []
            default_order_label = []
            for batch in evaluate_data_loader.graph_data_loader:
                print("evaluating data loader ..............")
                batch_pivot_node_features = batch.pivot_node_features.to(
                    self.device)
                batch_children_node_features = batch.children_node_features.to(
                    self.device)
                batch_edge_indexes = batch.edge_index.to(self.device)
                cur_default_order_label = batch.label.numpy()

                with torch.no_grad():
                    predict_scores = self.policy(
                        batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
                predict_scores = predict_scores.cpu().detach().numpy()
                predict_scores_list.append(predict_scores)
                default_order_label.append(cur_default_order_label)
            predict_scores = np.vstack(predict_scores_list)
            default_order_label = np.vstack(default_order_label)
            predict_scores = np.mean(predict_scores, axis=1, keepdims=True)
            ascending_indexes = np.argsort(
                predict_scores.squeeze())  # default ascending order
            print(f"predict_scores: {predict_scores}")
            self.tb_logger.add_histogram(
                'predict scores', predict_scores, global_step=epoch)
            # accuracy
            num_total_samples = default_order_label.shape[0]
            # accuracy = self.get_accuracy(
            #     predict_scores, default_order_label, num_total_samples)
            # precision and recall
            print(f"before confusion matrix scores: {predict_scores}")
            tp, tn, fp, fn, num_predict_positive_indexes = self.get_confusion_matrix(
                predict_scores, default_order_label)
            precision = tp / (tp + fp + 1e-3)
            recall = tp / (tp + fn + 1e-3)

            # if self.get_positive_scores(predict_scores):
            #     self.tb_logger.add_histogram('predict scores positive', self.get_positive_scores(predict_scores), global_step=epoch)
            # top k accuracy
            positive_indexes = np.nonzero(default_order_label)[0]
            num_positive_indexes = positive_indexes.shape[0]
            # with torch.no_grad():
            #     ascending_indexes, predict_scores = self.policy.inference(
            #         batch_pivot_node_features,
            #         batch_edge_indexes,
            #         batch_children_node_features
            #     )
            total_num = num_total_samples
            cnt_top_k = [0 for _ in range(9)]
            for i in range(9):
                percent = 0.1 * (i+1)
                sel_num = int(total_num * percent)
                sel_indexes = ascending_indexes[total_num-sel_num:]
                cnt = 0
                inter_indexes = list(set(sel_indexes).intersection(positive_indexes))
                cnt = len(inter_indexes)
                # for index in positive_indexes:
                #     if index in sel_indexes:
                #         cnt += 1
                cnt_top_k[i] = cnt

            # self.tb_logger.add_scalar(
            #     F'accuracy {npy_file}', accuracy, global_step=epoch)
            self.tb_logger.add_scalar(
                f'precision {npy_file}', precision, global_step=epoch)
            self.tb_logger.add_scalar(
                f'recall {npy_file}', recall, global_step=epoch)
            self.tb_logger.add_scalar(
                f'num predict postive {npy_file}', num_predict_positive_indexes / num_total_samples, global_step=epoch)
            for i in range(9):
                self.tb_logger.add_scalar(
                    f'top {i+1}0% accuracy {npy_file}', cnt_top_k[i] / num_positive_indexes, global_step=epoch)

    def train(self):
        # cross entropy loss
        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)
            print(f"*******epoch {epoch}, loss {loss.item()}*******")
            # if self.evaluate_data_loader is None:
            #     self.evaluate(epoch, loss)
            # else:
            #     self.evaluate_graph(epoch, loss)

    def save_checkpoint(self, itr):
        # TODO: add value net params save checkpoint
        model_state_dict = self.policy.state_dict()
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        save_model_f = os.path.join(self.model_save_dir, f'itr_{itr}.pkl')
        torch.save(model_state_dict, save_model_f)


class MultiClassTrainer(Trainer):
    def train_epoch(self, epoch):
        loss_all = 0
        cnt = 0
        for batch in self.data_loader.graph_data_loader:
            batch_pivot_node_features = batch.pivot_node_features.to(
                self.device)
            batch_children_node_features = batch.children_node_features.to(
                self.device)
            batch_edge_indexes = batch.edge_index.to(self.device)
            labels = batch.label.squeeze().to(self.device)
            # labels = F.one_hot(labels.squeeze())
            if self.debug_log:
                print(f"debug log one hot label shape: {labels.shape}")
                print(f"debug log one hot label: {labels}")
            predict_y, _ = self.policy(
                batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
            batch_loss = self.loss(predict_y, labels)
            print(f"epoch {epoch}, step {cnt+1}, loss {batch_loss.item()}")
            # update
            self.policy_optimizer.zero_grad()
            batch_loss.backward()
            self.policy_optimizer.step()

            loss_all += batch_loss
            cnt += 1
        if self.step % 1 == 0:
            if self.evaluate_data_loader is None:
                self.evaluate(self.step, batch_loss)
            else:
                self.evaluate_graph(self.step, batch_loss)
            self.save_checkpoint(self.step)
        self.step += 1
        return loss_all / cnt

    def _get_top_k_accuracy(self, predict_probabilities, labels, k):
        total_samples = labels.shape[0]
        num_classes = predict_probabilities.shape[1]
        cnt = 0
        # np.argsort ascending order
        sorted_indices = np.argsort(predict_probabilities, axis=1)
        if self.debug_log:
            print(f"debug log sorted indices shape: {sorted_indices.shape}")
            print(f"debug log sorted indices: {sorted_indices}")
            print(f"debug log true labels: {labels}")
        for i in range(total_samples):
            label = labels[i]
            indices = list(sorted_indices[i])
            if int(label) in indices[num_classes-k:]:
                cnt += 1

        return cnt / total_samples

    def evaluate_graph(self, epoch, loss):
        # log
        self.tb_logger.add_scalar('train loss', loss.item(), global_step=epoch)
        # true label
        for evaluate_data_loader, npy_file in self.evaluate_data_loader:
            for batch in evaluate_data_loader.graph_data_loader:
                print("evaluating data loader ..............")
                batch_pivot_node_features = batch.pivot_node_features.to(
                    self.device)
                batch_children_node_features = batch.children_node_features.to(
                    self.device)
                batch_edge_indexes = batch.edge_index.to(self.device)
                default_order_label = batch.label.numpy()
                with torch.no_grad():
                    _, predict_scores = self.policy(
                        batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
                predict_scores = predict_scores.cpu().detach().numpy()
                for k in range(3):
                    top_k_accuracy = self._get_top_k_accuracy(
                        predict_scores, default_order_label, k+1)
                    self.tb_logger.add_scalar(
                        F'top {k} accuracy {npy_file}', top_k_accuracy, global_step=epoch)


class MultiDomainTrainer(Trainer):
    def focal_loss(self, predict_y, target_y, data_loader):
        epilson = 1e-3
        alpha_positive = (data_loader.num_negative_samples+1e-3) / \
            (data_loader.num_positive_samples+data_loader.num_negative_samples)
        alpha_negative = (data_loader.num_positive_samples+1e-3) / \
            (data_loader.num_positive_samples+data_loader.num_negative_samples)
        # positive
        weight_positive = torch.pow(1.-predict_y+epilson, self.focal_gamma)
        focal_positive = -alpha_positive * \
            weight_positive * torch.log(predict_y+epilson)
        loss_positive = target_y * focal_positive
        # negative
        weight_negative = torch.pow(predict_y, self.focal_gamma)
        focal_negative = -alpha_negative * \
            weight_negative * torch.log(1.-predict_y+epilson)
        loss_negative = (1.-target_y) * focal_negative

        loss = torch.mean(loss_positive+loss_negative)
        return loss

    def process_loss(self, loss_all):
        if self.multi_loss == 'mean':
            loss = torch.mean(loss_all)
            indices = -1
        elif self.multi_loss == 'max':
            loss, indices = torch.max(loss_all, dim=0)
        elif self.multi_loss == 'soft_max':
            loss_sorted, indices = torch.sort(loss_all)  # ascending order
            loss = torch.mean(loss_sorted[self.drop_num:])
            indices = indices[self.drop_num]

        return loss, indices

    def train_epoch(self, epoch):
        num_domain = len(self.data_loader)
        loss_all = torch.zeros(num_domain)
        cnt = 0
        # self.policy.train()
        # assume that policy type == 'gcn'
        assert self.policy_type == 'gcn'
        for i, train_data_loader in enumerate(self.data_loader):
            for batch in train_data_loader.graph_data_loader:
                batch_pivot_node_features = batch.pivot_node_features.to(
                    self.device)
                batch_children_node_features = batch.children_node_features.to(
                    self.device)
                batch_edge_indexes = batch.edge_index.to(self.device)
                labels = batch.label.float().to(self.device)
                predict_y = self.policy(
                    batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
                predict_y = predict_y[:, i:i+1]
                batch_loss = self.loss(predict_y, labels, train_data_loader)
                loss_all[i] = batch_loss
                # self.policy_optimizer.zero_grad()
                # batch_loss.backward()
                # self.policy_optimizer.step()
                print(
                    f"cuda memory: {torch.cuda.memory_allocated(0)/1024**3} GB")
                print(f"cuda cached: {torch.cuda.memory_cached(0)/1024**3} GB")
                print(f"step: {i}, batch loss: {batch_loss.item()}")
                torch.cuda.empty_cache()
                break
        loss, indices = self.process_loss(loss_all)

        print(f"epoch {epoch}, step {cnt+1}, loss {loss.item()}")
        # update
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        # lr decay
        if self.use_lr_decay:
            self.lr_scheduler.step()
        cnt += 1
        if self.step % self.evaluate_freq == 0:
            if self.evaluate_data_loader is None:
                self.evaluate(self.step, batch_loss)
            else:
                # self.evaluate_graph(
                #     self.step, loss, indices=indices, batch_loss=loss_all)
                self.evaluate_graph(self.step, loss)
            self.save_checkpoint(self.step)
        self.step += 1
        return loss

    def evaluate_graph_backup(self, epoch, loss, indices=-1, batch_loss=[]):
        self.policy.eval()
        # log
        self.tb_logger.add_scalar('train loss', loss.item(), global_step=epoch)
        if indices != -1:
            self.tb_logger.add_scalar(
                'max loss indices', indices.item(), global_step=epoch)
        if batch_loss != []:
            for i in range(batch_loss.shape[0]):
                self.tb_logger.add_scalar(
                    f'batch loss {i}', batch_loss[i].item(), global_step=epoch)
        # true label
        for evaluate_data_loader, npy_file in self.evaluate_data_loader:
            for batch in evaluate_data_loader.graph_data_loader:
                print("evaluating data loader ..............")
                batch_pivot_node_features = batch.pivot_node_features.to(
                    self.device)
                batch_children_node_features = batch.children_node_features.to(
                    self.device)
                batch_edge_indexes = batch.edge_index.to(self.device)
                default_order_label = batch.label.numpy()

                with torch.no_grad():
                    predict_scores, node_embeddings = self.policy(
                        batch_pivot_node_features, batch_edge_indexes, batch_children_node_features, return_embedding=True)
                    if self.attention_model is not None:
                        _, attention_weights = self.attention_model(
                            batch_pivot_node_features, batch_edge_indexes, batch_children_node_features)
                    else:
                        attention_weights = None
                if attention_weights is not None:
                    predict_scores = predict_scores * attention_weights
                predict_scores = predict_scores.cpu().detach().numpy()
                for i in range(predict_scores.shape[1]):
                    self.tb_logger.add_histogram(f'predict scores {i+1}', predict_scores[:,i], global_step=epoch)
                predict_scores_std = np.std(
                    predict_scores, axis=1, keepdims=True)
                if self.policy.mean_max == 'max':
                    predict_scores = predict_scores.max(axis=1, keepdims=True)
                elif self.policy.mean_max == 'mean_std':
                    predict_scores = np.mean(
                        predict_scores, axis=1, keepdims=True) + np.std(predict_scores, axis=1, keepdims=True)
                elif self.policy.mean_max == 'mean':
                    predict_scores = np.mean(
                        predict_scores, axis=1, keepdims=True)
                elif self.policy.mean_max == 'attention':
                    assert self.attention_model is not None
                    attention_weights = attention_weights.cpu().detach().numpy()
                    # predict_scores = predict_scores * attention_weights
                    predict_scores = np.sum(
                        predict_scores, axis=1, keepdims=True)
                    print(f"attention weights: {attention_weights}")
                    self.tb_logger.add_histogram(
                        'attention weights', attention_weights, global_step=epoch)

                print(f"predict_scores: {predict_scores}")
                self.tb_logger.add_histogram(
                    'predict scores', predict_scores, global_step=epoch)

                # accuracy
                num_total_samples = default_order_label.shape[0]
                accuracy = self.get_accuracy(
                    predict_scores, default_order_label, num_total_samples)
                # precision and recall
                print(f"before confusion matrix scores: {predict_scores}")
                tp, tn, fp, fn, num_predict_positive_indexes = self.get_confusion_matrix(
                    predict_scores, default_order_label)
                precision = tp / (tp + fp + 1e-3)
                recall = tp / (tp + fn + 1e-3)

                # if self.get_positive_scores(predict_scores):
                #     self.tb_logger.add_histogram('predict scores positive', self.get_positive_scores(predict_scores), global_step=epoch)
                # top k accuracy
                positive_indexes = np.nonzero(default_order_label)[0]
                num_positive_indexes = positive_indexes.shape[0]
                with torch.no_grad():
                    ascending_indexes, predict_scores = self.policy.inference(
                        batch_pivot_node_features,
                        batch_edge_indexes,
                        batch_children_node_features,
                        attention=attention_weights
                    )
                    total_num = predict_scores.shape[0]
                    cnt_top_k = [0 for _ in range(9)]

                    for i in range(9):
                        percent = 0.1 * (i+1)
                        sel_num = int(total_num * percent)
                        sel_indexes = ascending_indexes[total_num-sel_num:]
                        cnt = 0
                        for index in positive_indexes:
                            if index in sel_indexes:
                                cnt += 1
                            else:
                                if i == 4:
                                    ascending_indexes_std = np.argsort(
                                        predict_scores_std.squeeze())
                                    ascending_indexes_std = list(
                                        ascending_indexes_std)
                                    wrong_sample_std_index = ascending_indexes_std.index(
                                        index)
                                    print(
                                        f"wrong_sample_std_index percent: {(len(ascending_indexes_std)-wrong_sample_std_index)/len(ascending_indexes_std)}")

                        cnt_top_k[i] = cnt

                self.tb_logger.add_scalar(
                    f'accuracy {npy_file}', accuracy, global_step=epoch)
                self.tb_logger.add_scalar(
                    f'precision {npy_file}', precision, global_step=epoch)
                self.tb_logger.add_scalar(
                    f'recall {npy_file}', recall, global_step=epoch)
                self.tb_logger.add_scalar(
                    f'num predict postive {npy_file}', num_predict_positive_indexes / num_total_samples, global_step=epoch)
                self.tb_logger.add_scalar(
                    f'num_positive_indexes {npy_file}', num_positive_indexes, global_step=epoch)
                # self.tb_logger.add_embedding(
                #     node_embeddings, global_step=epoch, tag=f"evaluate node embeddings {npy_file}"
                # )
                for i in range(9):
                    self.tb_logger.add_scalar(
                        f'top {i+1}0% accuracy {npy_file}', cnt_top_k[i] / num_positive_indexes, global_step=epoch)


class MultiDomainRobustTrainer(MultiDomainTrainer):
    def focal_loss(self, predict_y, target_y, data_loader):
        epilson = 1e-3
        alpha_positive = (data_loader.num_negative_samples+1e-3) / \
            (data_loader.num_positive_samples+data_loader.num_negative_samples)
        alpha_negative = (data_loader.num_positive_samples+1e-3) / \
            (data_loader.num_positive_samples+data_loader.num_negative_samples)
        # positive
        weight_positive = torch.pow(1.-predict_y+epilson, self.focal_gamma)
        focal_positive = -alpha_positive * \
            weight_positive * torch.log(predict_y+epilson)
        loss_positive = target_y * focal_positive
        # negative
        weight_negative = torch.pow(predict_y, self.focal_gamma)
        focal_negative = -alpha_negative * \
            weight_negative * torch.log(1.-predict_y+epilson)
        loss_negative = (1.-target_y) * focal_negative
        loss = torch.mean(loss_positive+loss_negative)

        # weighting domain loss
        # w_positive = torch.sum(weight_positive * target_y) / \
        #     data_loader.num_positive_samples
        w_positive = torch.mean(
            weight_positive * target_y + weight_negative * (1.-target_y))
        # w_negative = weight_negative * (1.-target_y)

        loss = w_positive * loss
        return loss


if __name__ == '__main__':
    pass
    # from o2_data_loader import DataLoader, GraphDataLoader
    # from score_policy import ScorePolicy
    # from gcn_policy import GCNPolicy
    # device = 'cuda:6'

    # train_data_loader = GraphDataLoader(
    #     './npy_data/v2/v12_data_train',
    #     'v12_processed_data_default_order',
    #     processed_npy_path=None,
    #     sample_bool=False,
    #     train_type='train',
    #     sample_type='upsampling',
    #     load_type='default_order',
    #     batch_size=512
    # )

    # # valid_data_loader = GraphDataLoader(
    # #     './npy_data/v2/arith/save_data_total_log2.blif_arith.npy',
    # #     processed_npy_path='./npy_data/v2/v1_processed_data/graph_samples_train_type_valid_sample_bool_False.npy',
    # #     sample_bool=False,
    # #     train_type='valid',
    # #     sample_type='upsampling',
    # #     load_type='default_order',
    # #     batch_size=512
    # # )

    # test_data_loader = GraphDataLoader(
    #     './npy_data/v2/v1_data_test/save_data_total_multiplier.blif_v1_data_test.npy',
    #     'v12_processed_data',
    #     processed_npy_path=None,
    #     sample_bool=False,
    #     train_type='test',
    #     sample_type='upsampling',
    #     load_type='default_order'
    # )

    # policy = GCNPolicy(
    #     emd_size = 128
    # )
    # policy = policy.to(device)
    # trainer = Trainer(
    #     train_data_loader,
    #     policy,
    #     evaluate_data_loader=test_data_loader,
    #     policy_type='gcn',
    #     lr=1e-4,
    #     epochs=10,
    #     device=device
    # )
    # trainer.train()

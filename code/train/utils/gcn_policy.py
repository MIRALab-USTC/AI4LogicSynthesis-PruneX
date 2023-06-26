import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        # print(f"n_units: {n_units}")
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
            self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BipartiteGraphConvolution2(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size=128):
        super().__init__('mean')

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features))
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
        # return self.output_module(torch.cat([output, right_features], dim=-1))

    def message(self, node_features_i, node_features_j):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_right(node_features_j))
        return output


class GCNPolicy(torch.nn.Module):
    def __init__(
        self,
        mean_max='mean',
        emd_size=128,
        out_size=2,
        num_pivot_node_features=69,
        num_children_node_features=69
    ):
        super().__init__()
        self.emd_size = emd_size
        self.num_pivot_node_features = num_pivot_node_features
        self.num_children_node_features = num_children_node_features
        self.out_size = out_size

        # pivot node EMBEDDING
        self.pivot_node_embedding = torch.nn.Sequential(
            PreNormLayer(self.num_pivot_node_features),
            torch.nn.Linear(self.num_pivot_node_features, emd_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emd_size, emd_size),
            torch.nn.ReLU(),
        )

        # VARIABLE EMBEDDING
        self.children_node_embedding = torch.nn.Sequential(
            PreNormLayer(self.num_children_node_features),
            torch.nn.Linear(self.num_children_node_features, emd_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emd_size, emd_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution2(emb_size=emd_size)
        self.conv_c_to_v = BipartiteGraphConvolution2(emb_size=emd_size)

        self.initial_output_module()

        self.mean_max = mean_max

    def initial_output_module(self):
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emd_size, self.emd_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emd_size, self.out_size),
            torch.nn.Sigmoid(),
        )

    def get_output(self, pivot_node_features):
        output = self.output_module(pivot_node_features)
        return output

    def forward(self, pivot_node_features, edge_indices, children_features, return_embedding=False):
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0)
        # print("before pivot node embedding")
        pivot_node_features = self.pivot_node_embedding(pivot_node_features)
        # print("after pivot node embedding")
        children_features = self.children_node_embedding(children_features)
        # print("after children node embedding")
        children_features = self.conv_v_to_c(
            pivot_node_features, edge_indices, children_features)
        pivot_node_features = self.conv_c_to_v(
            children_features, reversed_edge_indices, pivot_node_features)

        output = self.get_output(pivot_node_features)
        if return_embedding:
            return output, pivot_node_features
        return output

    def inference(self, pivot_node_features, edge_indices, children_features, attention=None):
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0)
        # print("before pivot node embedding")
        pivot_node_features = self.pivot_node_embedding(pivot_node_features)
        # print("after pivot node embedding")
        children_features = self.children_node_embedding(children_features)
        # print("after children node embedding")
        children_features = self.conv_v_to_c(
            pivot_node_features, edge_indices, children_features)
        pivot_node_features = self.conv_c_to_v(
            children_features, reversed_edge_indices, pivot_node_features)

        scores = self.get_output(pivot_node_features)
        scores = scores.cpu().detach().numpy()
        if self.mean_max is not None:
            std = np.std(scores, axis=1, keepdims=True)
            if attention is None:
                if self.mean_max == 'max':
                    scores = scores.max(axis=1, keepdims=True)
                elif self.mean_max == 'mean_std':
                    scores = np.mean(scores, axis=1, keepdims=True) + \
                        np.std(scores, axis=1, keepdims=True)
                elif self.mean_max == 'mean':
                    scores = np.mean(scores, axis=1, keepdims=True)
            else:
                scores = scores * attention
                scores = np.sum(scores, axis=1, keepdims=True)
        ascending_indexes = np.argsort(
            scores.squeeze())  # default ascending order

        return ascending_indexes, scores


class GCNMultiHeadPolicy(GCNPolicy):
    def __init__(
        self,
        mean_max='mean',
        emd_size=128,
        out_size=6,
        num_pivot_node_features=69,
        num_children_node_features=69
    ):
        GCNPolicy.__init__(
            self,
            mean_max=mean_max,
            emd_size=emd_size,
            out_size=out_size,
            num_pivot_node_features=num_pivot_node_features,
            num_children_node_features=num_children_node_features
        )

    def initial_output_module(self):
        self.output_module = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.emd_size, self.emd_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emd_size, 1),
                torch.nn.Sigmoid(),
            ) for _ in range(self.out_size)
        ])

    def get_output(self, pivot_node_features):
        output = [cur_module(pivot_node_features)
                  for cur_module in self.output_module]
        output = torch.cat(output, dim=1)

        return output


class GCNSoftmaxPolicy(GCNPolicy):
    def __init__(
        self,
        mean_max='mean',
        emd_size=128,
        out_size=6,
        num_pivot_node_features=69,
        num_children_node_features=69
    ):
        GCNPolicy.__init__(
            self,
            mean_max=mean_max,
            emd_size=emd_size,
            out_size=out_size,
            num_pivot_node_features=num_pivot_node_features,
            num_children_node_features=num_children_node_features
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emd_size, emd_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emd_size, out_size)
        )
        self.softmax_module = torch.nn.Softmax(dim=1)

    def forward(self, pivot_node_features, edge_indices, children_features):
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0)
        # print("before pivot node embedding")
        pivot_node_features = self.pivot_node_embedding(pivot_node_features)
        # print("after pivot node embedding")
        children_features = self.children_node_embedding(children_features)
        # print("after children node embedding")
        children_features = self.conv_v_to_c(
            pivot_node_features, edge_indices, children_features)
        pivot_node_features = self.conv_c_to_v(
            children_features, reversed_edge_indices, pivot_node_features)

        logits = self.output_module(pivot_node_features)
        probabilities = self.softmax_module(logits)

        return logits, probabilities


if __name__ == "__main__":
    """
    test for GCNPolicy
    """
    device = "cuda:0"
    gnn_policy = GCNPolicy().to(device)
    pivot_node_features = torch.randn((1, 69), device=device)
    children_features = torch.randn((3, 69), device=device)
    zero_row = np.zeros((1, 3), dtype=np.int64)
    one_row = np.arange(3).astype(np.int64)
    one_row = np.expand_dims(one_row, axis=0)
    edge_indexes = np.vstack([zero_row, one_row])
    edge_indexes = torch.tensor(edge_indexes, device=device)
    
    # first no train eval
    out = gnn_policy(
        pivot_node_features,
        edge_indexes,
        children_features
    )
    print(f"1: out {out}")

    # second train
    gnn_policy.train()
    out = gnn_policy(
        pivot_node_features,
        edge_indexes,
        children_features
    )    
    print(f"2: out {out}")
    # third eval
    gnn_policy.eval()
    out = gnn_policy(
        pivot_node_features,
        edge_indexes,
        children_features
    )
    print(f"3: out {out}")
    # second train
    gnn_policy.train()
    out = gnn_policy(
        pivot_node_features,
        edge_indexes,
        children_features
    )    
    print(f"4: out {out}")
    gnn_policy.eval()
    out = gnn_policy(
        pivot_node_features,
        edge_indexes,
        children_features
    )
    print(f"5: out {out}")
    """
    test for GCNMultiHeadPolicy
    """
    # device = "cuda:7"
    # gnn_policy = GCNMultiHeadPolicy().to(device)
    # pivot_node_features = torch.randn((1,69), device=device)
    # children_features = torch.randn((3,69), device=device)
    # zero_row = np.zeros((1,3), dtype=np.int64)
    # one_row = np.arange(3).astype(np.int64)
    # one_row = np.expand_dims(one_row, axis=0)
    # edge_indexes = np.vstack([zero_row, one_row])
    # edge_indexes = torch.tensor(edge_indexes, device=device)

    # out = gnn_policy(
    #     pivot_node_features,
    #     edge_indexes,
    #     children_features
    # )

    # print(f"out: {out}")
    # print(f"shape: {out.shape}")

    """
    test for GCNSoftmaxPolicy
    """
    # device = "cuda:7"
    # gnn_policy = GCNSoftmaxPolicy(mean_max='mean', out_size=6).to(device)
    # pivot_node_features = torch.randn((1,69), device=device)
    # children_features = torch.randn((3,69), device=device)
    # zero_row = np.zeros((1,3), dtype=np.int64)
    # one_row = np.arange(3).astype(np.int64)
    # one_row = np.expand_dims(one_row, axis=0)
    # edge_indexes = np.vstack([zero_row, one_row])
    # edge_indexes = torch.tensor(edge_indexes, device=device)

    # logits, probabilities = gnn_policy(
    #     pivot_node_features,
    #     edge_indexes,
    #     children_features
    # )
    # print(f"logits shape: {logits.shape}")
    # print(f"probabilities shape: {probabilities.shape}")
    # print(f"logits: {logits}")
    # print(f"probabilities: {probabilities}")

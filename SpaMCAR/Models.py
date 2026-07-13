import copy
import numpy as np
import random
import torch.nn.functional as F
import torch
from torch import nn
import copy  # import
from torch_geometric.nn import (
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

import faiss
import math

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,  # nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat": # Default heads=1
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin": # Default heads=1
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config.get('feat_hidden1', 64)
        self.feat_hidden2 = config.get('feat_hidden2', 32)
        self.gcn_hidden = config.get('gcn_hidden', 64)
        self.latent_dim = config.get('latent_dim', 16)
        self.p_drop = config.get('p_drop', 0.2)
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))
        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return x

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.get('latent_dim', 16)
        self.gcn_hidden = config.get('project_dim', 32)
        self.p_drop = config.get('p_drop', 0.2)
        self.layer1 = GraphConv(self.input_dim, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.layer2 = nn.Linear(self.gcn_hidden, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config.get('latent_dim', 16)
        self.p_drop = config.get('p_drop', 0.2)
        self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        return self.layer1(x, edge_index)


class Neighbor(nn.Module):
    def __init__(
        self,
        device,
        num_centroids,
        num_kmeans,
        clus_num_iters,
        use_prototype_triplet=False,
        negative_strategy='random',
        semi_hard_ratio=0.3,
        semi_hard_pool_size=64,
        use_semi_hard_schedule=False,
        semi_hard_ratio_start=0.1,
        semi_hard_ratio_end=0.5,
        false_negative_sim_threshold=0.95,
        boundary_aware_sampling=False,
        boundary_sampling_strength=1.0,
    ):
        super(Neighbor, self).__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters
        self.use_prototype_triplet = use_prototype_triplet
        self.negative_strategy = negative_strategy
        self.semi_hard_ratio = float(semi_hard_ratio)
        self.semi_hard_pool_size = int(semi_hard_pool_size)
        self.use_semi_hard_schedule = bool(use_semi_hard_schedule)
        self.semi_hard_ratio_start = float(semi_hard_ratio_start)
        self.semi_hard_ratio_end = float(semi_hard_ratio_end)
        self.current_semi_hard_ratio = float(semi_hard_ratio)
        self.false_negative_sim_threshold = float(false_negative_sim_threshold)
        self.boundary_aware_sampling = bool(boundary_aware_sampling)
        self.boundary_sampling_strength = float(boundary_sampling_strength)

    def set_epoch_progress(self, progress):
        if self.use_semi_hard_schedule:
            p = max(0.0, min(1.0, float(progress)))
            self.current_semi_hard_ratio = self.semi_hard_ratio_start + (self.semi_hard_ratio_end - self.semi_hard_ratio_start) * p
        else:
            self.current_semi_hard_ratio = self.semi_hard_ratio

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def _prototype_triplets(self, anchor, node_labels, student, teacher):
        unique_labels, inverse = torch.unique(node_labels, sorted=True, return_inverse=True)
        if unique_labels.numel() < 2:
            return None, None

        n_centers = unique_labels.numel()
        dim = teacher.size(1)
        centers = torch.zeros((n_centers, dim), device=self.device, dtype=teacher.dtype)
        centers.index_add_(0, inverse, teacher)
        counts = torch.bincount(inverse, minlength=n_centers).to(self.device).unsqueeze(1).clamp_min(1)
        centers = centers / counts

        anchor_cluster = inverse[anchor]
        positive_proto = centers[anchor_cluster]

        anchor_feat = student[anchor]
        center_sim = torch.matmul(anchor_feat, centers.t())
        row_idx = torch.arange(anchor.size(0), device=self.device)
        center_sim[row_idx, anchor_cluster] = -1e9
        neg_cluster = torch.argmax(center_sim, dim=1)
        negative_proto = centers[neg_cluster]
        return positive_proto, negative_proto

    def _compute_boundary_scores(self, node_labels, I_knn):
        """Estimate boundary uncertainty by neighborhood label inconsistency."""
        if I_knn.numel() == 0:
            return torch.ones_like(node_labels, dtype=torch.float32, device=self.device)
        n_nodes = node_labels.size(0)
        neigh_labels = node_labels[I_knn]
        self_labels = node_labels.unsqueeze(1).expand_as(neigh_labels)
        mismatch = (neigh_labels != self_labels).float()
        boundary_score = mismatch.mean(dim=1)
        # Keep all nodes sampleable; boundary nodes receive higher probability.
        return torch.clamp(boundary_score, min=0.0, max=1.0).reshape(n_nodes)

    def _boundary_resample_pairs(self, anchor, positive, boundary_scores):
        if anchor.numel() == 0:
            return anchor, positive
        strength = max(0.0, self.boundary_sampling_strength)
        weights = 1.0 + strength * boundary_scores[anchor]
        weights = torch.clamp(weights, min=1e-6)
        num_samples = int(anchor.size(0))
        prob = weights / torch.clamp(weights.sum(), min=1e-6)
        expected = prob * float(num_samples)

        # Deterministic weighted resampling via quota allocation.
        counts = torch.floor(expected).long()
        remain = num_samples - int(counts.sum().item())
        if remain > 0:
            fractional = expected - counts.float()
            top_idx = torch.argsort(fractional, descending=True)[:remain]
            counts[top_idx] += 1

        base_idx = torch.arange(num_samples, device=anchor.device, dtype=torch.long)
        sample_idx = torch.repeat_interleave(base_idx, counts)

        if sample_idx.numel() > num_samples:
            sample_idx = sample_idx[:num_samples]
        elif sample_idx.numel() < num_samples:
            pad_need = num_samples - sample_idx.numel()
            pad_src = torch.argsort(weights, descending=True)
            if pad_src.numel() == 0:
                pad = torch.zeros(pad_need, device=anchor.device, dtype=torch.long)
            else:
                repeat_times = (pad_need + pad_src.numel() - 1) // pad_src.numel()
                pad = pad_src.repeat(repeat_times)[:pad_need]
            sample_idx = torch.cat([sample_idx, pad], dim=0)

        return anchor[sample_idx], positive[sample_idx]

    def _sample_one_random_excluding(self, n_data, anchor_i, pos_set, similarity_row=None):
        if n_data <= 2:
            return (anchor_i + 1) % n_data
        for _ in range(16):
            cand = random.randint(0, n_data - 1)
            if cand == anchor_i or cand in pos_set:
                continue
            if similarity_row is not None and similarity_row[cand].item() >= self.false_negative_sim_threshold:
                continue
            if cand != anchor_i and cand not in pos_set:
                return cand
        for cand in range(n_data):
            if cand == anchor_i or cand in pos_set:
                continue
            if similarity_row is not None and similarity_row[cand].item() >= self.false_negative_sim_threshold:
                continue
            if cand != anchor_i and cand not in pos_set:
                return cand
        return (anchor_i + 1) % n_data

    def _sample_negative_indices(self, similarity, anchor, positive, n_data):
        anchor_cpu = anchor.detach().cpu().tolist()
        positive_cpu = positive.detach().cpu().tolist()

        pos_dict = {}
        for a, p in zip(anchor_cpu, positive_cpu):
            if a not in pos_dict:
                pos_dict[a] = set()
            pos_dict[a].add(p)

        negatives = []
        strategy = str(self.negative_strategy).lower()
        use_semi_hard = strategy == 'semi_hard'
        use_exclude_pos = strategy in ('exclude_pos', 'semi_hard')
        semi_hard_ratio = max(0.0, min(1.0, self.current_semi_hard_ratio))

        for a in anchor_cpu:
            pos_set = pos_dict.get(a, set())

            if use_semi_hard and random.random() < semi_hard_ratio:
                k = min(n_data, max(8, self.semi_hard_pool_size))
                cand_list = torch.topk(similarity[a], k=k, largest=True, sorted=True).indices.detach().cpu().tolist()
                chosen = None
                for cand in cand_list:
                    if cand == a or cand in pos_set:
                        continue
                    if similarity[a, cand].item() >= self.false_negative_sim_threshold:
                        continue
                    if cand != a and cand not in pos_set:
                        chosen = cand
                        break
                if chosen is not None:
                    negatives.append(chosen)
                    continue

            if use_exclude_pos:
                negatives.append(self._sample_one_random_excluding(n_data, a, pos_set, similarity_row=similarity[a]))
            else:
                negatives.append(random.randint(0, n_data - 1))

        return torch.tensor(negatives, device=self.device, dtype=torch.long)

    def forward(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:, 0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality
        ind = pos_.coalesce()._indices()
        anchor = ind[0]
        positive = ind[1]

        if self.boundary_aware_sampling and anchor.numel() > 0:
            node_labels = torch.mode(cluster_labels, dim=0).values
            boundary_scores = self._compute_boundary_scores(node_labels, I_knn)
            anchor, positive = self._boundary_resample_pairs(anchor, positive, boundary_scores)

        if self.use_prototype_triplet:
            # Use consensus labels from multiple kmeans runs as pseudo cluster ids.
            node_labels = torch.mode(cluster_labels, dim=0).values
            positive_proto, negative_proto = self._prototype_triplets(anchor, node_labels, student, teacher)
            if positive_proto is not None and negative_proto is not None:
                return anchor, positive_proto, negative_proto

        negative = self._sample_negative_indices(similarity, anchor, positive, n_data)
        return anchor, positive, negative

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result
class SMGA(nn.Module):
    def __init__(self, num_clusters, input_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.dec_in_dim = config.get('latent_dim', 16)
        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()
        self.projector = Projector(config)
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.decoder = Decoder(input_dim, config)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.base_use_prototype_triplet = config.get('use_prototype_triplet', False)
        self.prototype_triplet_start_epoch = int(config.get('prototype_triplet_start_epoch', 0))
        self.neighbor = Neighbor(
            device,
            num_clusters,
            config.get('num_kmeans', 5),
            config.get('clus_num_iters', 20),
            use_prototype_triplet=False,
            negative_strategy=config.get('negative_strategy', 'semi_hard'),
            semi_hard_ratio=config.get('semi_hard_ratio',0.3),
            semi_hard_pool_size=config.get('semi_hard_pool_size', 48),
            use_semi_hard_schedule=config.get('use_semi_hard_schedule', False),
            semi_hard_ratio_start=config.get('semi_hard_ratio_start', 0.1),
            semi_hard_ratio_end=config.get('semi_hard_ratio_end', 0.5),
            false_negative_sim_threshold=config.get('false_negative_sim_threshold', 0.95),
            boundary_aware_sampling=config.get('boundary_aware_sampling', True),
            boundary_sampling_strength=config.get('boundary_sampling_strength', 1.0),
        )
        self.topk = config.get('topk', 30)
        self.mask_rate = config.get('mask_rate', 0.5)
        self.t = config.get('t', 2)
        self.momentum_rate = config.get('momentum_rate', 0.998)
        self.use_momentum_schedule = config.get('use_momentum_schedule', False)
        self.momentum_start = float(config.get('momentum_start', self.momentum_rate))
        self.momentum_end = float(config.get('momentum_end', self.momentum_rate))
        self.momentum_schedule_type = config.get('momentum_schedule_type', 'cosine')
        self.current_momentum = self.momentum_rate
        self.replace_rate = config.get('replace_rate', 0.1)
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None
        self.adj_temperature = config.get('adj_temperature', 0.7)
        self.adj_loss_remove_self_loop = config.get('adj_loss_remove_self_loop', True)
        self.adj_loss_mode = config.get('adj_loss_mode', 'sampled')
        self.adj_neg_ratio = config.get('adj_neg_ratio', 2)
        self.adj_max_pos_edges = config.get('adj_max_pos_edges', 15000)
        self.adj_from_clean_view = config.get('adj_from_clean_view', True)
        self.adj_use_norm_emb = config.get('adj_use_norm_emb', True)
        self.adj_stop_grad = config.get('adj_stop_grad', False)
        self.adj_use_proj_head = config.get('adj_use_proj_head', True)
        self.adj_proj_hidden = config.get('adj_proj_hidden', 32)
        if self.adj_use_proj_head:
            self.adj_projector = nn.Sequential(
                nn.Linear(self.dec_in_dim, self.adj_proj_hidden),
                nn.PReLU(),
                nn.Linear(self.adj_proj_hidden, self.dec_in_dim),
            )

    def set_training_epoch(self, epoch, total_epochs=None):
        enable_proto = self.base_use_prototype_triplet and (int(epoch) >= self.prototype_triplet_start_epoch)
        self.neighbor.use_prototype_triplet = bool(enable_proto)
        progress = 0.0
        if total_epochs is not None and total_epochs > 1:
            progress = float(epoch) / float(max(1, int(total_epochs) - 1))
            progress = max(0.0, min(1.0, progress))
        if hasattr(self.neighbor, 'set_epoch_progress'):
            self.neighbor.set_epoch_progress(progress)
        if self.use_momentum_schedule and total_epochs is not None and total_epochs > 1:
            if self.momentum_schedule_type == 'linear':
                self.current_momentum = self.momentum_start + (self.momentum_end - self.momentum_start) * progress
            else:
                # cosine warm-up style: smooth transition from start to end
                cosine_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
                self.current_momentum = self.momentum_start + (self.momentum_end - self.momentum_start) * cosine_scale
        else:
            self.current_momentum = self.momentum_rate

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0.1):
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def _build_adj_target(self, adj_label, device):
        if adj_label is None:
            return None
        if adj_label.is_sparse:
            adj_target = adj_label.to_dense().float().to(device)
        else:
            adj_target = adj_label.float().to(device)

        if self.adj_loss_remove_self_loop:
            adj_target = adj_target.clone()
            adj_target.fill_diagonal_(0)
        return adj_target

    def _sample_negative_edges(self, adj_target, num_neg):
        n_nodes = adj_target.size(0)
        neg_i = []
        neg_j = []
        remain = num_neg

        # Rejection sampling on upper-triangle pairs to avoid duplicate undirected edges.
        while remain > 0:
            draw_size = max(remain * 2, 1024)
            i = torch.randint(0, n_nodes, (draw_size,), device=adj_target.device)
            j = torch.randint(0, n_nodes, (draw_size,), device=adj_target.device)
            i_min = torch.minimum(i, j)
            j_max = torch.maximum(i, j)

            valid = i_min < j_max
            valid = valid & (adj_target[i_min, j_max] < 0.5)

            if valid.any():
                i_valid = i_min[valid][:remain]
                j_valid = j_max[valid][:remain]
                neg_i.append(i_valid)
                neg_j.append(j_valid)
                remain -= i_valid.numel()

        neg_i = torch.cat(neg_i, dim=0)
        neg_j = torch.cat(neg_j, dim=0)
        return neg_i, neg_j

    def _dense_adj_recon_loss(self, emb, adj_target):
        logits = torch.matmul(emb, emb.t()) / self.adj_temperature

        if self.adj_loss_remove_self_loop:
            eye_mask = torch.eye(adj_target.size(0), device=adj_target.device, dtype=torch.bool)
            logits = logits.masked_fill(eye_mask, 0.0)

        pos_count = adj_target.sum().clamp(min=1.0)
        total_count = torch.tensor(adj_target.numel(), device=adj_target.device, dtype=adj_target.dtype)
        neg_count = (total_count - adj_target.sum()).clamp(min=1.0)
        pos_weight = (neg_count / pos_count).detach()
        return F.binary_cross_entropy_with_logits(logits, adj_target, pos_weight=pos_weight)

    def _sampled_adj_recon_loss(self, emb, adj_target):
        pos_edges = torch.nonzero(adj_target > 0.5, as_tuple=False)
        if pos_edges.numel() == 0:
            return torch.zeros(1, device=emb.device).squeeze(0)

        pos_edges = pos_edges[pos_edges[:, 0] < pos_edges[:, 1]]
        if pos_edges.numel() == 0:
            return torch.zeros(1, device=emb.device).squeeze(0)

        if pos_edges.size(0) > self.adj_max_pos_edges:
            keep = torch.randperm(pos_edges.size(0), device=emb.device)[:self.adj_max_pos_edges]
            pos_edges = pos_edges[keep]

        num_pos = pos_edges.size(0)
        num_neg = max(1, int(num_pos * self.adj_neg_ratio))
        neg_i, neg_j = self._sample_negative_edges(adj_target, num_neg)

        pos_i = pos_edges[:, 0]
        pos_j = pos_edges[:, 1]
        pos_logits = (emb[pos_i] * emb[pos_j]).sum(dim=-1) / self.adj_temperature
        neg_logits = (emb[neg_i] * emb[neg_j]).sum(dim=-1) / self.adj_temperature

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
        return 0.5 * (pos_loss + neg_loss)

    def adj_recon_loss(self, emb, adj_label):
        if adj_label is None:
            return torch.zeros(1, device=emb.device).squeeze(0)

        adj_target = self._build_adj_target(adj_label, emb.device)
        if adj_target is None:
            return torch.zeros(1, device=emb.device).squeeze(0)
        if self.adj_loss_mode == 'dense':
            return self._dense_adj_recon_loss(emb, adj_target)
        return self._sampled_adj_recon_loss(emb, adj_target)

    def _prepare_adj_embedding(self, x, edge_index, masked_emb):
        if self.adj_from_clean_view:
            emb = self.online_encoder(x, edge_index)
        else:
            emb = masked_emb

        if self.adj_use_proj_head:
            emb = self.adj_projector(emb)

        if self.adj_use_norm_emb:
            emb = F.normalize(emb, dim=-1, p=2)

        if self.adj_stop_grad:
            emb = emb.detach()
        return emb

    def mask_attr_prediction(self, x, edge_index, flag, adj_label=None):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)
        enc_rep = self.online_encoder(use_x, use_adj)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            enc_rep_t = self.target_encoder(x_t, use_adj)
            rep_t = enc_rep_t
            self.momentum_update(self.current_momentum)

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.
        rep[mask_nodes] += self.rep_mask
        rep = self.projector(rep, use_adj)

        online = rep[mask_nodes]
        target = rep_t[mask_nodes]

        rep[keep_nodes] = 0.0
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        if flag:
            self.anchor_pair = self.neighbor(
                use_adj,
                F.normalize(enc_rep, dim=-1, p=2),
                F.normalize(enc_rep_t, dim=-1, p=2),
                self.topk,
            )
        if self.anchor_pair is not None:
            anchor, positive, negative = self.anchor_pair
            tri_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
        else:
            tri_loss = torch.zeros(1, device=enc_rep.device).squeeze(0)

        mean_loss = F.mse_loss(online, target)
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)
        adj_emb = self._prepare_adj_embedding(x, edge_index, enc_rep)
        adj_loss = self.adj_recon_loss(adj_emb, adj_label)

        return mean_loss, rec_loss, tri_loss, adj_loss

    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        if positive.dtype in (torch.int32, torch.int64) and positive.dim() == 1:
            positive_arr = emb[positive]
        else:
            positive_arr = positive
        if negative.dtype in (torch.int32, torch.int64) and negative.dim() == 1:
            negative_arr = emb[negative]
        else:
            negative_arr = negative
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, flag, adj_label=None):
        return self.mask_attr_prediction(x, edge_index, flag, adj_label=adj_label)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        enc_rep = self.online_encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon
# class SMGA(nn.Module):
#     def __init__(self, num_clusters, input_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
#         super().__init__()
#         self.dec_in_dim = config['latent_dim']
#         self.online_encoder = Encoder(input_dim, config)
#         self.target_encoder = copy.deepcopy(self.online_encoder)
#         self._init_target()
#         self.projector = Projector(config)
#         self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
#         nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
#         self.decoder = Decoder(input_dim, config)
#         self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
#         self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
#         self.neighbor = Neighbor(device, num_clusters, config['num_kmeans'], config['clus_num_iters'])
#         self.topk = config['topk']
#         self.mask_rate = config['mask_rate']
#         self.t = config['t']
#         self.momentum_rate = config['momentum_rate']
#         self.replace_rate = config['replace_rate']
#         self.mask_token_rate = 1 - self.replace_rate
#         self.anchor_pair = None

#     def _init_target(self):
#         for param_teacher in self.target_encoder.parameters():
#             param_teacher.detach()
#             param_teacher.requires_grad = False

#     def momentum_update(self, base_momentum=0.1):
#         for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
#             param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

#     def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
#         num_nodes = x.shape[0]
#         perm = torch.randperm(num_nodes, device=x.device)
#         num_mask_nodes = int(mask_rate * num_nodes)
#         mask_nodes = perm[: num_mask_nodes]
#         keep_nodes = perm[num_mask_nodes:]

#         if self.replace_rate > 0:
#             num_noise_nodes = int(self.replace_rate * num_mask_nodes)
#             perm_mask = torch.randperm(num_mask_nodes, device=x.device)
#             token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
#             noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
#             noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

#             out_x = x.clone()
#             out_x[token_nodes] = 0.0
#             out_x[noise_nodes] = x[noise_to_be_chosen]

#         else:
#             out_x = x.clone()
#             token_nodes = mask_nodes
#             out_x[mask_nodes] = 0.0

#         out_x[token_nodes] += self.enc_mask_token
#         use_edge_index = edge_index.clone()

#         return out_x, use_edge_index, (mask_nodes, keep_nodes)

#     def mask_attr_prediction(self, x, edge_index, flag):
#         use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)
#         enc_rep = self.online_encoder(use_x, use_adj)

#         with torch.no_grad():
#             x_t = x.clone()
#             x_t[keep_nodes] = 0.0
#             x_t[keep_nodes] += self.enc_mask_token
#             enc_rep_t = self.target_encoder(x_t, use_adj)
#             rep_t = enc_rep_t
#             self.momentum_update(self.momentum_rate)

#         rep = enc_rep
#         rep = self.encoder_to_decoder(rep)
#         rep[mask_nodes] = 0.
#         rep[mask_nodes] += self.rep_mask
#         rep = self.projector(rep, use_adj)

#         online = rep[mask_nodes]
#         target = rep_t[mask_nodes]

#         rep[keep_nodes] = 0.0
#         recon = self.decoder(rep, use_adj)
#         x_init = x[mask_nodes]
#         x_rec = recon[mask_nodes]

#         if flag:
#             self.anchor_pair = self.neighbor(use_adj, F.normalize(enc_rep, dim=-1, p=2), F.normalize(enc_rep_t, dim=-1, p=2), self.topk)
#         if self.anchor_pair is not None:
#             anchor, positive, negative = self.anchor_pair
#             tri_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
#         else:
#             tri_loss = 0

#         mean_loss = F.mse_loss(online, target)
#         rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

#         return mean_loss, rec_loss, tri_loss


#     def sce_loss(self, x, y, t=2):
#         x = F.normalize(x, p=2, dim=-1)
#         y = F.normalize(y, p=2, dim=-1)
#         cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
#         loss = -torch.log(cos_m.pow_(t))
#         return loss.mean()

#     def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
#         anchor_arr = emb[anchor]
#         positive_arr = emb[positive]
#         negative_arr = emb[negative]
#         triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
#         tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
#         return tri_output

#     def forward(self, x, edge_index, flag):
#         return self.mask_attr_prediction(x, edge_index, flag)

#     @torch.no_grad()
#     def evaluate(self, x, edge_index):
#         enc_rep = self.online_encoder(x, edge_index)
#         rep = self.encoder_to_decoder(enc_rep)
#         rep = self.projector(rep, edge_index)
#         recon = self.decoder(rep, edge_index)
#         return enc_rep, recon


class EmbeddingFusionLayer(nn.Module):
    def __init__(self, input_dim, conv_emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv_emb_dim = conv_emb_dim
        self.conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=conv_emb_dim, kernel_size=1)
        
    def _index_position_encoding(self, num_nodes, d_model, device):
        """"""
        position = torch.arange(0, num_nodes, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        position_encoding = torch.zeros(num_nodes, d_model, device=device)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding
    # def our_position_encoding(self, num_nodes, input_dim, device):
    #     """
    #     Args:
    #         num_nodes: 
    #         input_dim: 
    #         device: 
    #     Returns:
    #         position_encoding: [num_nodes, input_dim]
    #     """
    #     position_encoding = torch.zeros(num_nodes, input_dim, device=device)
    #     position = torch.arange(0, num_nodes, device=device).unsqueeze(1).float()  # [num_nodes, 1]
        
    #     # 
    #     # 
    #     half_dim = input_dim // 2
        
    #     #  div_term half_dim
    #     div_term = torch.exp(torch.arange(0, half_dim, device=device).float() * 
    #                         -(math.log(10000.0) / half_dim))
        
    #     # 
    #     # position: [num_nodes, 1], div_term: [half_dim]
    #     # : [num_nodes, half_dim]
    #     sin_values = torch.sin(position * div_term)
    #     cos_values = torch.cos(position * div_term)
        
    #     # 
    #     if input_dim % 2 == 0:
    #         #  input_dim 
    #         position_encoding[:, 0::2] = sin_values
    #         position_encoding[:, 1::2] = cos_values
    #     else:
    #         #  input_dim 
    #         position_encoding[:, 0:2*half_dim:2] = sin_values
    #         position_encoding[:, 1:2*half_dim:2] = cos_values
    #         # 
    #         position_encoding[:, -1] = torch.sin(position.squeeze() * div_term[-1])
        
    #     return position_encoding


    def _spatial_position_encoding(self, spatial_coords, d_model, device):
        """"""
        coords = spatial_coords
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32, device=device)
        else:
            coords = coords.to(device=device, dtype=torch.float32)

        if coords.dim() != 2 or coords.size(1) < 2:
            raise ValueError("spatial_coords must have shape [num_nodes, >=2]")

        coords = coords[:, :2]
        coords = (coords - coords.mean(dim=0, keepdim=True)) / (coords.std(dim=0, keepdim=True) + 1e-8)

        n_freq = max(1, d_model // 4)
        if n_freq == 1:
            freq = torch.ones(1, device=device)
        else:
            freq = torch.exp(torch.linspace(0, -math.log(10000.0), n_freq, device=device))

        x_coord = coords[:, 0:1]
        y_coord = coords[:, 1:2]
        x_enc = torch.cat([torch.sin(x_coord * freq), torch.cos(x_coord * freq)], dim=1)
        y_enc = torch.cat([torch.sin(y_coord * freq), torch.cos(y_coord * freq)], dim=1)
        position_encoding = torch.cat([x_enc, y_enc], dim=1)

        if position_encoding.size(1) < d_model:
            pad = torch.zeros(position_encoding.size(0), d_model - position_encoding.size(1), device=device)
            position_encoding = torch.cat([position_encoding, pad], dim=1)
        elif position_encoding.size(1) > d_model:
            position_encoding = position_encoding[:, :d_model]
        return position_encoding


    def forward(self, x, spatial_coords=None):
        num_nodes = x.size(0)
        device = x.device
        
        x_reshaped = x.unsqueeze(0).permute(0, 2, 1)  # [1, input_dim, num_nodes]
        conv_emb = self.conv_layer(x_reshaped).squeeze(0).permute(1, 0)  # [num_nodes, conv_emb_dim]
        
        if spatial_coords is None:
            position_embedding = self._index_position_encoding(num_nodes, self.input_dim, device)
        else:
            position_embedding = self._spatial_position_encoding(spatial_coords, self.input_dim, device)
        
        x_combined = torch.cat([x, conv_emb, position_embedding], dim=-1)  # [num_nodes, input_dim+conv_emb_dim+input_dim]
        
        return x_combined

class FusedSMGA(SMGA):
    def __init__(self, num_clusters, input_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        conv_emb_dim = config.get('conv_emb_dim', 64)
        combined_dim = 2 * input_dim + conv_emb_dim
        
        super().__init__(num_clusters, combined_dim, config, device)
        
        self.embedding_fusion = EmbeddingFusionLayer(input_dim, conv_emb_dim)
        
        self.online_encoder = Encoder(combined_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()
        
        self.decoder = Decoder(input_dim, config)
        
    def mask_attr_prediction(self, x, edge_index, flag, spatial_coords=None, adj_label=None):
        x_combined = self.embedding_fusion(x, spatial_coords)
        
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x_combined, edge_index, self.mask_rate)
        enc_rep = self.online_encoder(use_x, use_adj)
        
        with torch.no_grad():
            x_t = x_combined.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            enc_rep_t = self.target_encoder(x_t, use_adj)
            rep_t = enc_rep_t
            self.momentum_update(self.current_momentum)
        
        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.
        rep[mask_nodes] += self.rep_mask
        rep = self.projector(rep, use_adj)
        
        online = rep[mask_nodes]
        target = rep_t[mask_nodes]
        
        rep[keep_nodes] = 0.0
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]  # 
        x_rec = recon[mask_nodes]
        
        if flag:
            self.anchor_pair = self.neighbor(
                use_adj,
                F.normalize(enc_rep, dim=-1, p=2),
                F.normalize(enc_rep_t, dim=-1, p=2),
                self.topk,
            )
        if self.anchor_pair is not None:
            anchor, positive, negative = self.anchor_pair
            tri_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
        else:
            tri_loss = torch.zeros(1, device=enc_rep.device).squeeze(0)
        
        mean_loss = F.mse_loss(online, target)
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)
        adj_emb = self._prepare_adj_embedding(x_combined, edge_index, enc_rep)
        adj_loss = self.adj_recon_loss(adj_emb, adj_label)
        
        return mean_loss, rec_loss, tri_loss, adj_loss

    def forward(self, x, edge_index, flag, spatial_coords=None, adj_label=None):
        return self.mask_attr_prediction(x, edge_index, flag, spatial_coords, adj_label=adj_label)

    @torch.no_grad()
    def evaluate(self, x, edge_index, spatial_coords=None):
        x_combined = self.embedding_fusion(x, spatial_coords)
        
        enc_rep = self.online_encoder(x_combined, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon
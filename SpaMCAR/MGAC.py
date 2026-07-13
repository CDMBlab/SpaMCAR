# MGAC.py - Masked Graph Autoencoder with Contrastive Augmentation
import os
from tqdm import tqdm
from torch.backends import cudnn
import random
import numpy as np
import torch
from SpaMCAR.Models import SMGA, FusedSMGA

class Mgac:
    def __init__(self, adata, graph_dict, num_clusters, device, config, roundseed=0):
        seed = config.get('seed', 3407)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.graph_dict = graph_dict
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters

    def _start_(self):
        # Use multi-scale features instead of raw PCA features
        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['multi_scale_features'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception

        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]

        if 'spatial' in self.adata.obsm:
            spatial_arr = np.asarray(self.adata.obsm['spatial'])
            if spatial_arr.ndim == 2 and spatial_arr.shape[1] >= 2:
                self.spatial_coords = torch.FloatTensor(spatial_arr[:, :2].copy()).to(self.device)
            else:
                self.spatial_coords = None
                print("Warning: spatial coords shape is invalid, fallback to index position encoding.")
        else:
            self.spatial_coords = None
            print("Warning: adata.obsm['spatial'] not found, fallback to index position encoding.")

        self.input_dim = self.X.shape[-1]
        # Fix config variable reference issue
        if self.model_config.get('use_enhanced', False):
            print("Using FusedSMGA model")
            self.model = FusedSMGA(self.num_clusters, self.input_dim, self.model_config, self.device).to(self.device)
        else:
            print("Using SMGA model")
            self.model = SMGA(self.num_clusters, self.input_dim, self.model_config, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _fit_(self):
        # Initialize learning rate scheduler
        scheduler_config = self.train_config.get('lr_scheduler', 'none')
        if scheduler_config == 'cosine':
            scheduler = torch.optim.lr.CosineAnnealingLR(
                self.optimizer, T_max=self.train_config['epochs'], eta_min=1e-6
            )
        elif scheduler_config == 'step':
            scheduler = torch.optim.lr.StepLR(
                self.optimizer, step_size=100, gamma=0.5
            )
        else:
            scheduler = None


        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            if hasattr(self.model, 'set_training_epoch'):
                self.model.set_training_epoch(epoch, self.train_config['epochs'])
            flag = False
            if epoch % self.train_config['t_step'] == 0:
                flag = True

            w_recon = self.train_config['w_recon']
            w_mean = self.train_config['w_mean']
            w_tri = self.train_config['w_tri']
            w_adj_target = self.train_config.get('w_adj', 0.0)
            adj_start_epoch = self.train_config.get('adj_start_epoch', 0)
            adj_warmup_epochs = max(1, self.train_config.get('adj_warmup_epochs', 1))
            if epoch < adj_start_epoch:
                w_adj = 0.0
            elif epoch < adj_start_epoch + adj_warmup_epochs:
                warmup_progress = float(epoch - adj_start_epoch + 1) / float(adj_warmup_epochs)
                w_adj = w_adj_target * warmup_progress
            else:
                w_adj = w_adj_target

            if self.model_config.get('use_enhanced', False):
                mean_loss, rec_loss, tri_loss, adj_loss = self.model(
                    self.X,
                    self.adj_norm,
                    flag,
                    spatial_coords=self.spatial_coords,
                    adj_label=self.adj_label,
                )
            else:
                mean_loss, rec_loss, tri_loss, adj_loss = self.model(
                    self.X,
                    self.adj_norm,
                    flag,
                    adj_label=self.adj_label,
                )
            loss = w_recon * rec_loss + w_mean * mean_loss + w_tri * tri_loss + w_adj * adj_loss
            loss.backward()

            grad_clip = self.train_config.get('gradient_clipping', 5.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    grad_clip
                )

            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()


            if epoch == self.train_config['epochs'] - 1:
                torch.save(self.model.state_dict(), 'best_model.pth')

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Epoch {epoch} loss={loss:.3f} recon={rec_loss:.3f} mean={mean_loss:.3f} tri={tri_loss:.3f} adj={adj_loss:.3f} w_adj={w_adj:.3f} lr={current_lr:.6f}",
                refresh=True)

    def trian(self):
        self._start_()
        self._fit_()

    def process(self):
        self.model.eval()
        if self.model_config.get('use_enhanced', False):
            enc_rep, recon = self.model.evaluate(self.X, self.adj_norm, self.spatial_coords)
        else:
            enc_rep, recon = self.model.evaluate(self.X, self.adj_norm)
        return enc_rep, recon

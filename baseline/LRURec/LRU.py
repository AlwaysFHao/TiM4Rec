# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/10/15
import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class LRURec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LRURec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.hidden_dropout_prob = config["dropout_prob"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.encoder = LRUModel(hidden_size=self.hidden_size,
                                n_layers=self.n_layers,
                                dropout_prob=self.hidden_dropout_prob)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.bias = torch.nn.Parameter(torch.zeros(self.n_items))
        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.truncated_normal_init()

    @staticmethod
    def _init_weights(module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    @staticmethod
    def right_to_left_padding(x) -> torch.Tensor:
        # x should be of shape [batch, seq_len]
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional [batch, seq_len].")

        batch, seq_len = x.size()

        # Calculate the number of non-zero elements (i.e. the length of each sequence without padding)
        non_zero_count = (x != 0).sum(dim=1)

        # Create a tensor filled with zeros
        result = torch.zeros_like(x)

        for i in range(batch):
            num_non_zeros = non_zero_count[i]
            if num_non_zeros > 0:
                # Copy non-zero elements to the rightmost positions of the result tensor
                result[i, -num_non_zeros:] = x[i, :num_non_zeros]

        return result

    @staticmethod
    def get_mask(x):
        return x > 0

    def forward(self, item_seq):
        item_seq = self.right_to_left_padding(item_seq)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        input_emb = self.dropout(input_emb)
        input_emb = self.layer_norm(input_emb)

        mask = self.get_mask(item_seq)

        output = self.encoder(input_emb, mask)
        return output[:, -1, :]  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.bias
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding(test_item)
        test_item_bias = self.bias[test_item]
        scores = (torch.mul(seq_output, test_item_emb) + test_item_bias).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output = self.forward(item_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.bias  # [B n_items]
        return scores


class LRUModel(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, dropout_prob: float):
        super(LRUModel, self).__init__()
        self.lru_blocks = nn.ModuleList([LRUBlock(hidden_size, dropout_prob) for _ in range(n_layers)])

    def forward(self, x, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)
        return x


class LRUBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float):
        super(LRUBlock, self).__init__()
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=dropout_prob)
        self.feed_forward = PositionWiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size * 4, dropout=dropout_prob)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x


class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    @staticmethod
    def lru_parallel(i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], dim=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu

        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)

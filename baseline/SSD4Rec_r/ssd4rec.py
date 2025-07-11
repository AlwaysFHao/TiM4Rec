# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/11/29

import math
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from ssd import SSD

"""
Refer from Mamba4Rec, Recbole-SASRec and Mamba
Mamba4Rec: https://github.com/chengkai-liu/Mamba4Rec
Mamba: https://github.com/state-spaces/mamba
"""


class SSD4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        """
        TiM4Rec
        Y = N * FFN(TiSSD(X, T))
        """
        super(SSD4Rec, self).__init__(config, dataset)

        # Hyperparameters for TiM4Rec
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        # Hyperparameters for SSDLayer
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']
        self.head_dim = config['head_dim']
        self.chunk_size = config['chunk_size']
        self.is_ffn = config['is_ffn']
        self.norm_eps = config['norm_eps']
        self.is_kai_ming_init = config['is_kai_ming_init']

        assert (self.hidden_size * self.expand) % self.head_dim == 0, \
            f'hidden_size * expand {self.hidden_size * self.expand} can\'t divisible by head_dim {self.head_dim} !'
        self.n_heads = (self.hidden_size * self.expand) // self.head_dim
        # [PAD] has been added to the actual number of items in Recbole
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.in_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)

        self.dropout = nn.Dropout(self.dropout_prob)

        self.ssd_layers = nn.ModuleList([
            SSDLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                is_ffn=self.is_ffn,
                norm_eps=self.norm_eps
            ) for _ in range(self.num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            if self.is_kai_ming_init:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.in_layer_norm(item_emb)
        for i in range(self.num_layers):
            item_emb = self.ssd_layers[i](item_emb)
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        # seq_output: [batch_size, hidden_size], test_item_emb: [num_items, hidden_size]
        logits = torch.einsum('bh,nh->bn', seq_output, test_item_emb)
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        """
        A prediction score is calculated for a given batch of sequences and a single candidate item for the
        corresponding sequence.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding(test_item)
        # seq_output: [batch_size, hidden_size], test_item_emb: [batch_size, hidden_size]
        scores = torch.einsum('bh,bh->b', seq_output, test_item_emb)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        # seq_output: [batch_size, hidden_size], test_item_emb: [num_items, hidden_size]
        scores = torch.einsum('bh,nh->bn', seq_output, test_item_emb)
        return scores


class SSDLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 num_layers: int,
                 head_dim: int,
                 chunk_size: int,
                 dropout: float,
                 is_ffn: bool = True,
                 norm_eps: float = 1e-12):
        """
        A single-layer SSDLayer, containing a SSDBlock and an FFN(if is_ffn is True)

        :param d_model: vector embedding dimension
        :param d_state: the B, C matrix dimension in SSD
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param num_layers: the number of SSDLayer layers,
                used to determined whether the SSDLayer needs residuals connections
        :param head_dim: Header dimension of an SSD
        :param chunk_size: Chunk size of an SSD
        :param dropout: dropout_radio
        :param is_ffn: whether the FFN is included
        :param norm_eps: normalization epsilon
        """
        super(SSDLayer, self).__init__()
        self.num_layers = num_layers
        self.ssd = SSD(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            norm_eps=norm_eps
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = FeedForward(
                d_model=d_model,
                inner_size=d_model * 4,
                dropout=dropout
            )

    def forward(self, x):
        """
        x -> ssd(x)
        -> ffn(x) if is_ffn is True
        :param x: shape: [batch_size, seq_len, d_model]
        :return: shape: [batch_size, seq_len, d_model]
        """
        # hidden = self.layer_norm(x)
        hidden = self.ssd(x)
        # Determine whether SSDBlock needs residual by num_layers
        if self.num_layers == 1:
            hidden = self.layer_norm(self.dropout(hidden))
        else:
            hidden = self.layer_norm(self.dropout(hidden) + x)

        if self.is_ffn:
            return self.ffn(hidden)
        else:
            return hidden


def gelu(x):
    """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2, norm_eps=1e-12):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Hardswish()
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.layer_norm(self.dropout(hidden) + x)
        return hidden


if __name__ == '__main__':
    pass

# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/6/13
import math
import torch
from einops import repeat
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from ssd import TiSSD

"""
Refer from Mamba4Rec, Recbole-SASRec and Mamba
Mamba4Rec: https://github.com/chengkai-liu/Mamba4Rec
Mamba: https://github.com/state-spaces/mamba
"""


class TiM4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        """
        TiM4Rec
        Y = N * FFN(TiSSD(X, T))
        """
        super(TiM4Rec, self).__init__(config, dataset)

        # Hyperparameters for TiM4Rec
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.time_drop_out = config['time_drop_out']

        # Hyperparameters for SSDLayer
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']
        self.head_dim = config['head_dim']
        self.chunk_size = config['chunk_size']
        self.is_ffn = config['is_ffn']
        self.is_time = config['is_time']
        self.p2p_residual = config['p2p_residual']
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
        if self.is_time:
            # self.time_start_token = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.layer_norm_time = nn.LayerNorm(self.max_seq_length, eps=self.norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.ssd_layers = nn.ModuleList([
            TiSSDLayer(
                d_model=self.hidden_size,
                seq_len=self.max_seq_length,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                time_drop_out=self.time_drop_out,
                is_ffn=self.is_ffn,
                is_time=self.is_time,
                p2p_residual=self.p2p_residual,
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

    def calculate_time_diff(self, time_stamp):
        """
        Calculate the interaction time difference
        :param time_stamp: [batch_size, seq_len]
        :return: [batch_size, seq_len]
        """
        batch_size = time_stamp.shape[0]
        # [batch_size, seq_len - 1]
        time_diff = time_stamp[:, 1:] - time_stamp[:, :-1]
        # add first time diff
        # time_diff = torch.concat([repeat(self.time_start_token, '1 -> b 1', b=batch_size), time_diff], dim=1)
        time_diff = torch.concat([torch.zeros(batch_size, 1).to(time_diff.device), time_diff], dim=1)
        # time_diff = nn.functional.normalize(time_diff, p=2, dim=-1)
        time_diff = self.layer_norm_time(self.dropout(time_diff))
        # [batch_size, seq_len] -> [batch_size, n_heads, seq_len]
        time_diff = repeat(time_diff, 'b l -> b h l', h=self.n_heads)
        return time_diff

    def forward(self, item_seq, item_seq_len, time_stamp):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.in_layer_norm(item_emb)
        if self.is_time:
            time_diff = self.calculate_time_diff(time_stamp)
        else:
            time_diff = None
        for i in range(self.num_layers):
            item_emb, time_diff = self.ssd_layers[i](item_emb, time_diff)
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamp = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, time_stamp)
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
        time_stamp = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, time_stamp)

        test_item_emb = self.item_embedding(test_item)
        # seq_output: [batch_size, hidden_size], test_item_emb: [batch_size, hidden_size]
        scores = torch.einsum('bh,bh->b', seq_output, test_item_emb)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_stamp = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, time_stamp)

        test_item_emb = self.item_embedding.weight
        # seq_output: [batch_size, hidden_size], test_item_emb: [num_items, hidden_size]
        scores = torch.einsum('bh,nh->bn', seq_output, test_item_emb)
        return scores


class TiSSDLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 num_layers: int,
                 head_dim: int,
                 chunk_size: int,
                 dropout: float,
                 time_drop_out: float,
                 is_ffn: bool = True,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12):
        """
        A single-layer TiSSDLayer, containing a TiSSDBlock and an FFN(if is_ffn is True)

        :param d_model: vector embedding dimension
        :param d_state: the B, C matrix dimension in SSD
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param num_layers: the number of SSDLayer layers,
                used to determined whether the SSDLayer needs residuals connections
        :param head_dim: Header dimension of an SSD
        :param chunk_size: Chunk size of an SSD
        :param dropout: dropout_radio
        :param time_drop_out: time_dropout_radio
        :param is_ffn: whether the FFN is included
        :param is_time: whether the Time-aware is included
        :param p2p_residual: whether you use point-to-point residuals
        :param norm_eps: normalization epsilon
        """
        super(TiSSDLayer, self).__init__()
        self.num_layers = num_layers
        self.ssd = TiSSD(
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            time_drop_out=time_drop_out,
            is_time=is_time,
            p2p_residual=p2p_residual,
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

    def forward(self, x, time_diff):
        """
        x -> ssd(x)
        -> ffn(x) if is_ffn is True
        :param x: shape: [batch_size, seq_len, d_model]
        :param time_diff: shape: [batch_size, seq_len]
        :return: shape: [batch_size, seq_len, d_model]
        """
        # hidden = self.layer_norm(x)
        hidden, time_diff = self.ssd(x, time_diff)
        # Determine whether SSDBlock needs residual by num_layers
        if self.num_layers == 1:
            hidden = self.layer_norm(self.dropout(hidden))
        else:
            hidden = self.layer_norm(self.dropout(hidden) + x)

        if self.is_ffn:
            return self.ffn(hidden), time_diff
        else:
            return hidden, time_diff


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

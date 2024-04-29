from torch import nn as nn
from bert_attention import *
import math
from collections import defaultdict,Counter
import numpy as np

class BERT(nn.Module):
    def __init__(self, args,device,original_data):
        super().__init__()
        self.args = args
        self.device = device 
        self.num_item = len(np.unique(original_data[:,1]))
        self.random_seed = np.random.randint(0,2147483647)
        self.embedding = BERTEmbedding(self.args,self.num_item)
        self.model = BERTModel(self.args,self.num_item)
        self.truncated_normal_init()
        self.original_data = original_data

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.model.named_parameters():
                if not 'layer_norm' in n:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, x):
        x, mask = self.embedding(x)
        scores = self.model(x, self.embedding.token.weight, mask)
        return scores

class BERTEmbedding(nn.Module):
    def __init__(self, args,num_item):
        super().__init__()
        vocab_size = num_item + 2
        hidden = args.bert_hidden_units
        max_len = args.bert_max_len
        dropout = args.bert_dropout

        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=hidden)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=hidden)

        self.layer_norm = LayerNorm(features=hidden)
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, x):
        if len(x.shape) > 2:
            x = torch.ones(x.shape[:2]).to(x.device)
        return (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

    def forward(self, x):
        mask = self.get_mask(x)
        if len(x.shape) > 2:
            pos = self.position(torch.ones(x.shape[:2]).to(x.device))
            x = torch.matmul(x, self.token.weight) + pos
        else:
            x = self.token(x) + self.position(x)
        return self.dropout(self.layer_norm(x)), mask


class BERTModel(nn.Module):
    def __init__(self, args,num_item):
        super().__init__()
        hidden = args.bert_hidden_units
        heads = args.bert_num_heads
        head_size = args.bert_head_size
        dropout = args.bert_dropout
        attn_dropout = args.bert_attn_dropout
        layers = args.bert_num_blocks

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            hidden, heads, head_size, hidden * 4, dropout, attn_dropout) for _ in range(layers)])
        self.linear = nn.Linear(hidden, hidden)
        self.bias = torch.nn.Parameter(torch.zeros(num_item + 2))
        self.bias.requires_grad = True
        self.activation = GELU()

    def forward(self, x, embedding_weight, mask):
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        x = self.activation(self.linear(x))
        scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
        return scores

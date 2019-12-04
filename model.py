import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

class DecoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        dropout,
        num_layers,
        attn_size=50,
        pointer=True,
        device='cuda'
    ):
        super(DecoderAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pointer = pointer
        self.device = device

        self.embeddingN = nn.Embedding(vocab_sizeN, embedding_sizeN, vocab_sizeN - 1)
        self.embeddingT = nn.Embedding(vocab_sizeT + attn_size + 3, embedding_sizeT, vocab_sizeT - 1)

        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.W_context = nn.Linear(
            embedding_sizeN + embedding_sizeT + hidden_size,
            hidden_size
        )
        self.dropout = nn.AlphaDropout(dropout)
        # self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstm = nn.LSTMCell(embedding_sizeN + embedding_sizeT, hidden_size)
        self.lstm = nn.LSTM(
            embedding_sizeN + embedding_sizeT,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.w_global = nn.Linear(hidden_size * 3, vocab_sizeT + attn_size + 3) # map into T
        if self.pointer:
            self.w_switcher = nn.Linear(hidden_size * 2, 1)
        self.selu = nn.SELU()

    def hook_hc(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_size).to(self.device)),
                    Variable(torch.zeros(batch_size, self.hidden_size).to(self.device)))

    def forward(
        self,
        input,
        hc,
        enc_out,
        mask,
        hc_parent
    ):
        n_input, t_input = input
        batch_size = n_input.size(0)

        # (enc_out, enc_out_W) [(batch_size, max_length, hidden_size * 2), (batch_size, max_length, hidden_size)]
        # mask (batch_size, max_length)
        # hidden_prev (batch_size, hidden_size)
        if hc is not None:
            h, c = hc
        else:
            h, c = self.hook_hc(batch_size)

        n_input = self.embeddingN(n_input)
        t_input = self.embeddingT(t_input)
        input = torch.cat([n_input, t_input], 1)
        input = self.dropout(input) # (batch_size, embedding_size)

        h, c = self.lstm(input, hc)

        scores = self.W_hidden(h).unsqueeze(1) # (batch_size, max_length, hidden_size)
        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(2) # (batch_size, max_length)
        scores = scores.masked_fill(mask, -1e20) # (batch_size, max_length)
        attn_weights = F.softmax(scores, dim=1) # (batch_size, max_length)
        attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1,  max_length)
        context = torch.matmul(attn_weights, enc_out).squeeze(1) # (batch_size, hidden_size)

        context = torch.cat((input, context), 1)

        hidden_attn = self.selu(self.W_context(context))

        # h, c = self.lstm(hidden_attn, hc) # (batch_size, 1,  hidden_size)

        w_t = F.log_softmax(self.w_global(torch.cat([h, c, hc_parent], dim=1)), dim=1)
        if self.pointer:
            s_t = F.sigmoid(self.w_switcher(torch.cat([h, c], dim=1)))
            return [s_t * w_t, (1 - s_t) * attn_weights.squeeze(1)], (h, c)
        else:
            return w_t, (h, c)

class MixtureAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_sizeT,
        vocab_sizeN,
        embedding_sizeT,
        embedding_sizeN,
        num_layers,
        dropout,
        device='cuda',
#         teacher_forcing_ratio = 0.7,
        label_smoothing = 0.1,
        pointer=True,
        attn_size=50,
        SOS_token=0
    ):
        super(MixtureAttention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.eof_N_id = vocab_sizeN - 1
        self.eof_T_id = vocab_sizeT - 1
        self.unk_id = vocab_sizeT - 2
        self.SOS_token = SOS_token
        self.attn_size = attn_size
        self.vocab_sizeT = vocab_sizeT
        self.vocab_sizeN = vocab_sizeN

        self.W_out = nn.Linear(hidden_size * 2, hidden_size)

        self.decoder = DecoderAttention(
            hidden_size=hidden_size,
            vocab_sizeT=vocab_sizeT,
            vocab_sizeN=vocab_sizeN,
            embedding_sizeT=embedding_sizeT,
            embedding_sizeN=embedding_sizeN,
            num_layers=num_layers,
            attn_size=attn_size,
            dropout=dropout,
            pointer=pointer,
            device=device
        ).to(device)

        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing,
                tgt_vocab_size=vocab_sizeT + attn_size + 3,
                ignore_index=self.eof_T_id,
                device=self.device
            ) # ignore EOF ?!
        else:
            self.criterion = nn.NLLLoss(reduction='none')

        self.pointer = pointer


    def forward(
        self,
        n_tensor,
        t_tensor,
        p_tensor
    ):
        batch_size = n_tensor.size(0)
        max_length = n_tensor.size(1)

        full_mask = (n_tensor == self.eof_N_id)

        input = (
            torch.ones(
                batch_size,
                dtype=torch.long,
                device=self.device
            ) * self.SOS_token,
            torch.ones(
                batch_size,
                dtype=torch.long,
                device=self.device
            ) * self.SOS_token
        )
        hs = torch.zeros(
            batch_size,
            max_length,
            self.hidden_size,
            requires_grad=False
        ).to(self.device)
        hc = None

        parent = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=self.device
        )

        loss = torch.tensor(0.0, device=self.device)

        token_losses = torch.zeros(
            batch_size,
            max_length
        ).to(self.device)

        ans = []

        for iter in range(max_length):
            memory = hs[:, iter - self.attn_size : iter]
            output, hc = self.decoder(
                input,
                hc,
                memory.clone().detach(),
                full_mask[:, iter - self.attn_size : iter],
                hs[torch.arange(batch_size),parent].squeeze(1).clone().detach()
            )
            hs[:, iter] = hc[0]
            if self.pointer:
                output, local_attn = output
                global_topv, global_topi = output.topk(1)
                if local_attn.shape[1] > 0:
                    local_topv, local_topi = local_attn.topk(1)
                    # replace global distribution with local distribution
                    for i in range(output.shape[0]):
                        if global_topv[i] > local_topv[i]:
                            output[i] = 0
                            output[i, self.vocab_sizeT:self.vocab_sizeT + len(local_attn[i])] = local_attn[i]
            topv, topi = output.topk(1)
            input = (n_tensor[:, iter].clone(), t_tensor[:, iter].clone())
            parent = p_tensor[:, iter]

            ans.append(topi.detach())
#                 cond = (t_tensor[:, iter] < self.vocab_sizeT + self.attn_size).long()
#                 masked_target = cond * t_tensor[:, iter] + (1 - cond) * self.eof_T_id
            token_losses[:, iter] = self.criterion(output, t_tensor[:, iter].clone().detach())

        loss = token_losses.sum() #/ batch_size
        return loss, torch.cat(ans, dim=1)

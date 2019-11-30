import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        dropout,
        PAD_token=1,
        vocab_size=80000,
        pointer=True
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, PAD_token)

        self.W_hidden = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.W_context = nn.Linear(
            embedding_size + hidden_size * 2,
            hidden_size
        )
        self.dropout = nn.AlphaDropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.w_global = nn.Linear(hidden_size * 3, vocab_size)
        if self.pointer:
            self.w_switcher = nn.Linear(hidden_size * 2, 1)
        self.selu = nn.SELU()

    def forward(
        self,
        input,
        hc,
        enc_out,
        mask,
        hc_parent
    ):
        batch_size = input.size(0)

        # (enc_out, enc_out_W) [(batch_size, max_length, hidden_size * 2), (batch_size, max_length, hidden_size)]
        # mask (batch_size, max_length)
        # hidden_prev (batch_size, hidden_size)

        h, c = hc

        input = self.embedding(input)
        input = self.dropout(input) # (batch_size, embedding_size)

        scores = self.W_hidden(h).unsqueeze(1) # (batch_size, max_length, hidden_size)
        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(2) # (batch_size, max_length)
        scores = scores.masked_fill(mask, -1e20) # (batch_size, max_length)
        attn_weights = F.softmax(scores, dim=1) # (batch_size, max_length)
        attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1,  max_length)
        context = torch.matmul(attn_weights, enc_out).squeeze(1) # (batch_size, hidden_size)

        context = torch.cat((input, context), 1)

        hidden_attn = self.selu(self.W_context(context))

        h, c = self.lstm(hidden_attn, hc)

        w_t = F.log_softmax(self.w_global(torch.cat([h, c, hc_parent[0]], dim=1)), dim=1)
        if self.pointer:
            s_t = F.sigmoid(self.w_switcher(torch.cat([h, c], dim=1)))
            return [s_t * w_t, (1 - s_t) * scores], (h, c)
        else:
            return w_t, (h, c)

class Seq2seq(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        num_layers,
        dropout,
        UNK_token = 0,
        PAD_token = 1,
        SOS_token = 2,
        EOS_token = 3,
        vocab_size = 80000,
        device='cuda',
        teacher_forcing_ratio = 0.7,
        label_smoothing = 0.1,
        pointer=True,
        attn_size=100
    ):
        super(Seq2seq, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            PAD_token=PAD_token,
            vocab_size=vocab_size
        ).to(device)

        self.W_out = nn.Linear(hidden_size * 2, hidden_size)

        self.hook_h = nn.Linear(
            hidden_size * 2,
            hidden_size
        ).to(device)

        self.hook_c = nn.Linear(
            hidden_size * 2,
            hidden_size
        ).to(device)

        self.decoder = Decoder(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
            PAD_token = PAD_token,
            vocab_size = vocab_size,
            pointer=pointer
        ).to(device)

        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=vocab_size, ignore_index=PAD_token)
        else:
            self.criterion = nn.NLLLoss(reduction='none')

        self.PAD_token = PAD_token
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.UNK_token = UNK_token
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.pointer = pointer

    def forward(
        self,
        s_tensor,
        parents_tensor
    ):
        batch_size = s_tensor.size(0)
        max_length = s_tensor.size(1)

        full_mask = (s_tensor == self.PAD_token)

        input = torch.ones(
            batch_size
            , dtype=torch.long
            , device=self.device
        ) * self.SOS_token

        hc = self.generate_input()

        hs = [hс[0]]

        if self.training:
            loss = torch.tensor(0.0, device=self.device)

            token_losses = torch.zeros(
                batch_size,
                max_length
            ).to(self.device)

            finished_indexes = torch.zeros(batch_size).bool().to(self.device)

            ans = []

            for iter in range(max_length - 1):
                memory = hs[:, iter - self.attn_size : iter]
                output, hc = self.decoder(
                    input,
                    hc,
                    memory,
                    full_mask[:, iter - self.attn_size : iter],
                    hs[parents_tensor[iter]]
                )
                hs.append(hc[0])
                if self.pointer:
                    output, local_attn = output
                    global_topv, global_topi = local_attn.topk(1)
                    local_topv, local_topi = local_attn.topk(1)
                    # replace global distribution with local distribution
                    for i in output.shape[0]:
                        if global_topv[i] > local_topv[i]:
                            output[i] = 0
                            output[i, :local_attn.shape[1]] = local_attn[i]

                topv, topi = output.topk(1)
                if np.random.rand() < self.teacher_forcing_ratio:
                    input = s_tensor[:, iter]
                else:
                    input = topi.squeeze(1).detach()

                ans.append(topi.detach())

                token_losses[~finished_indexes, iter] = self.criterion(output[~finished_indexes], s_tensor[~finished_indexes, iter + 1])
                finished_indexes |= (s_tensor[:, iter + 1] == self.EOS_token)

                # check if finish all
                if (finished_indexes.sum() == batch_size):
                    break

            loss = token_losses.sum() / batch_size
            return loss, torch.cat(ans, dim=1)

        else:
            # TODO: fix eval:
            
            #assert s.size(0) == 1

            ans = []
            # print(input.size())
            finished_indexes = torch.zeros(batch_size).bool().to(self.device)

            for iter in range(max_length - 1):
                memory = hs[:, iter - self.attn_size : iter]
                output, hc = self.decoder(
                    input,
                    hc,
                    memory,
                    full_mask[:, iter - self.attn_size : iter],
                    hs[parents_tensor[iter]]
                )
                hs.append(hc[0])
                if self.pointer:
                    output, local_attn = output
                    global_topv, global_topi = local_attn.topk(1)
                    local_topv, local_topi = local_attn.topk(1)
                    # replace global distribution with local distribution
                    for i in output.shape[0]:
                        if global_topv[i] > local_topv[i]:
                            output[i] = 0
                            output[i, :local_attn.shape[1]] = local_attn[i]

                topv, topi = output.topk(1)

                input = topi.squeeze(1).detach()

                ans.append(topi.detach())

                finished_indexes |= (topi.squeeze(1) == self.EOS_token)

                # check if finish all
                if (finished_indexes.sum() == batch_size):
                    print('all finished:', iter)
                    break


            return torch.cat(ans, dim=1)

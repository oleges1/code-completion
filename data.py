import torch
from preprocess_utils.utils import *

def fix_parent(p, start_i):
    p -= start_i
    return 0 if p < 0 else p

def data_gen(data, split_size):
    for sample in data:
        accum = []
        start_i = 0
        for i, item in enumerate(sample):
            n, t, p = item
            p = fix_parent(p, start_i)
            accum.append((n, t, p))
            if len(accum) == split_size:
                yield accum
                accum = []
                start_i = i
        if len(accum) > 0:
            yield accum

class MainDataset(torch.utils.data.Dataset):
    def __init__(self,
            N_filename = './pickle_data/PY_non_terminal_small.pickle',
            T_filename = './pickle_data/PY_terminal_10k_whole.pickle',
            is_train=False,
            truncate_size=150
        ):
        super(MainDataset).__init__()
        train_dataN, test_dataN, vocab_sizeN, train_dataT, test_dataT, vocab_sizeT, attn_size, train_dataP, test_dataP = input_data(
    N_filename, T_filename
)
        self.is_train = is_train
        if self.is_train:
            self.data = [item for item in data_gen(zip(train_dataN, train_dataT, train_dataP), truncate_size)]
        else:
            self.data = [item for item in data_gen(zip(test_dataN, test_dataT, test_dataP), truncate_size)]
        self.data = sorted(self.data, key=lambda x: len(x[0]))
        self.vocab_sizeN = vocab_sizeN
        self.vocab_sizeT = vocab_sizeT
        self.attn_size = attn_size
        self.eof_N_id = vocab_sizeN - 1
        self.eof_T_id = vocab_sizeT - 1
        self.unk_id = vocab_sizeT - 2
        self.truncate_size = truncate_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent_N, sent_T, sent_P = self.data[idx]
        if len(sent_N) > self.truncate_size:
            start_i = np.random.choice(len(sent_N) - self.truncate_size)
            sent_N = sent_N[start_i: start_i + self.truncate_size]
            sent_T = sent_T[start_i: start_i + self.truncate_size]
            sent_P = np.array(sent_P[start_i: start_i + self.truncate_size], dtype='int')
            sent_P -= start_i
            cond = np.array(sent_P < 0, dtype='int')
            sent_P = cond * 0 + sent_P * (1 - cond)
            sent_P = list(map(int, sent_P))
        return (sent_N, sent_T, sent_P)

    def collate_fn(self, samples, device='cpu'):
        sent_N = [sample[0] for sample in samples]
        sent_T = [sample[1] for sample in samples]
        sent_P = [sample[2] for sample in samples]

        s_max_length = max(map(lambda x: len(x), sent_N))

        sent_N_tensors = []
        sent_T_tensors = []
        sent_P_tensors = []

        for sn, st, sp in zip(sent_N, sent_T, sent_P):
            sn_tensor = torch.ones(
                s_max_length
                , dtype=torch.long
                , device=device
            ) * self.eof_N_id

            st_tensor = torch.ones(
                s_max_length
                , dtype=torch.long
                , device=device
            ) * self.eof_T_id

            sp_tensor = torch.ones(
                s_max_length
                , dtype=torch.long
                , device=device
            ) * 1

            for idx, w in enumerate(sn):
                sn_tensor[idx] = w
                st_tensor[idx] = st[idx]
                sp_tensor[idx] = sp[idx]
            sent_N_tensors.append(sn_tensor.unsqueeze(0))
            sent_T_tensors.append(st_tensor.unsqueeze(0))
            sent_P_tensors.append(sp_tensor.unsqueeze(0))

        sent_N_tensors = torch.cat(sent_N_tensors, dim=0)
        sent_T_tensors = torch.cat(sent_T_tensors, dim=0)
        sent_P_tensors = torch.cat(sent_P_tensors, dim=0)

        return sent_N_tensors, sent_T_tensors, sent_P_tensors

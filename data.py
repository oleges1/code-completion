import torch

def convert_to_ix(sent, field):
    res = []
    if isinstance(sent, str):
        sent = sent.split()
    for item in ['SOS'] + sent + ['EOS']:
        res.append(field.vocab.stoi[item])
    return res

def convert_to_sent(ix, field):
    res = []
    for item in ix:
        try:
            if field.vocab.itos[item] not in ['EOS', 'SOS', 'PAD']:
                res.append(field.vocab.itos[item])
            if field.vocab.itos[item] in ['EOS', 'PAD']:
                break
        except IndexError:
            break
    if len(res) == 0:
        res = ['PAD']
    return ' '.join(res)

class MainDataset(torch.utils.data.Dataset):
    def __init__(self, pickle_data, src_field):
        super(MainDataset).__init__()
        self.field = Field()
        self.src_field = src_field
        self.data = sorted(mt_data, key=lambda x: len(x.src))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_raw = self.data[idx].src
        trg_raw = self.data[idx].trg
        return convert_to_ix(src_raw, self.field), convert_to_ix(trg_raw, self.field)

def collate_fn(samples, PAD_token=1, device='cpu'):
    sent = [sample[0] for sample in samples]
    s_parent = [sample[1] for sample in samples]

    s_max_length = max(map(lambda x: len(x), sent))

    s_tensors = []
    s_tensor_parents = []

    for s, sp in zip(sent, s_parent):
        s_res = torch.ones(
            s_max_length
            , dtype=torch.long
            , device=device
        ) * PAD_token

        s_parent = torch.ones(
            s_max_length
            , dtype=torch.long
            , device=device
        ) * PAD_token

        for idx, w in enumerate(s):
            s_res[idx] = w
            s_parent[idx] = sp[idx]

        s_tensors.append(s_res.unsqueeze(0))
        s_tensor_parents.append(s_parent.unsqueeze(0))

    s_tensor = torch.cat(s_tensors, dim=0)
    s_tensor_parent = torch.cat(s_tensor_parents, dim=0)

    return s_tensor, s_tensor_parent

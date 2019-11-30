from model import *
from torchtext import datasets
from torchtext.data import Field
from data import *
import sacrebleu
import os

# config:
device = 'cuda'
batch_size = 32
vocab_size = 50000
LOAD_EPOCH = None
epochs = 1000
model_name = 'one_layer_dropout01'
hidden_size = 1024
embedding_size = 300
dropout = 0.1
num_layers = 1
lr = 0.001
num_workers = 6
teacher_forcing_ratio = 0.9
label_smoothing = 0.1

try:
    from tqdm import tqdm
    wrap = tqdm
except:
    wrap = lambda x: x

def train():
    src = Field(init_token = "SOS", eos_token = "EOS", pad_token="PAD")
    trg = Field(init_token = "SOS", eos_token = "EOS", pad_token="PAD")

    mt_train = datasets.TranslationDataset(
        path='homework_machine_translation_de-en/train.de-en', exts=('.de', '.en'),
        fields=(src, trg))

    mt_val = datasets.TranslationDataset(
        path='homework_machine_translation_de-en/val.de-en', exts=('.de', '.en'),
        fields=(src, trg))

    src.build_vocab(mt_train, mt_val, max_size=vocab_size)
    trg.build_vocab(mt_train, mt_val, max_size=vocab_size)

    PAD_token = 1
    assert PAD_token == src.vocab.stoi['PAD']
    assert PAD_token == trg.vocab.stoi['PAD']

    train_dataset = MainDataset(mt_train, src, trg)
    test_dataset = MainDataset(mt_val, src, trg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    s2s = Seq2seq(
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        num_layers=num_layers,
        dropout=dropout,
        vocab_size=vocab_size + 4, # SOS, EOS, PAD, UNK
        SOS_token=src.vocab.stoi['SOS'],
        EOS_token=trg.vocab.stoi['EOS'],
        teacher_forcing_ratio=teacher_forcing_ratio,
        label_smoothing=label_smoothing
    )
    start_epoch = 0
    if LOAD_EPOCH is not None:
        cpk = torch.load('checkpoints/%s/epoch_%04d.pth' % (model_name, LOAD_EPOCH))
        s2s.load_state_dict(cpk['model'])
        s2s = s2s.to(device)
        optimizer = torch.optim.AdamW(s2s.parameters(), lr=lr)
        optimizer.load_state_dict(cpk['optimizer'])
        start_epoch = cpk['epoch']
        print('loaded', start_epoch, '!')
    else:
        s2s = s2s.to(device)
        optimizer = torch.optim.AdamW(s2s.parameters(), lr=lr)

    for epoch in range(start_epoch, epochs):
        print("epoch: %04d" % epoch)
        loss_avg = 0
        total = len(train_loader)

        s2s = s2s.train()
        for i, (s, p) in enumerate(wrap(train_loader)):
            s, p = s.to(device), p.to(device)
            optimizer.zero_grad()

            loss, ans = s2s(s, p)
            loss_avg += loss.item()
            loss.backward()

            if (i + 1) % 100 == 0:
                print('temp_loss: %f' % (loss.item()))

            optimizer.step()

        print("avg_loss: %f" % (loss_avg/total))

        if (epoch+1) % 1 == 0:
            s2s = s2s.eval()
            bleu = 0.
            for i, (s, t, l) in enumerate(wrap(test_loader)):
                s, t, l = s.to(device), t, l.to(device)
                ans = s2s(s, l)
                pred_t, ref_t = [], []
                for pred, ref in zip(ans, t):
                    pred_t = [convert_to_sent(pred.detach().cpu().numpy(), trg)]
                    ref_t = [[convert_to_sent(ref.cpu().numpy(), trg)]]
                    bleu += sacrebleu.corpus_bleu(pred_t, ref_t).score
                if (i + 1) % 10 == 0:
                    print('\ntranslated:', pred_t[0], '\ntrue:', ref_t[0][0])

            bleu /= len(test_loader) * batch_size

            print("[metrics] bleu: %f" % (bleu))

        os.system('mkdir -p checkpoints/' + model_name)
        torch.save({
            'model': s2s.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, 'checkpoints/%s/epoch_%04d.pth' % (model_name, epoch))

if __name__ == '__main__':
    train()

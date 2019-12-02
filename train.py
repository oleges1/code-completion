from model import *
from data import *
import os
from tqdm import tqdm
import yaml
from utils import DotDict
from sklearn.metrics import accuracy_score
import torch

# config:
CONFIG_FILE = 'configs/default.yml'

def train(config):
    device = config.train.device
    
    data_train = MainDataset(
        N_filename = config.data.N_filename,
        T_filename = config.data.T_filename,
        is_train=True,
        truncate_size=config.data.truncate_size
    )
    
    data_val = MainDataset(
        N_filename = config.data.N_filename,
        T_filename = config.data.T_filename,
        is_train=False,
        truncate_size=config.data.truncate_size
    )

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=data_train.collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=data_val.collate_fn
    )

    model = MixtureAttention(
        hidden_size = config.model.hidden_size,
        vocab_sizeT = data_train.vocab_sizeT,
        vocab_sizeN = data_train.vocab_sizeN,
        attn_size = data_train.attn_size,
        embedding_sizeT = config.model.embedding_sizeT,
        embedding_sizeN = config.model.embedding_sizeN,
        num_layers = 1,
        dropout = config.model.dropout,
        label_smoothing = config.model.label_smoothing,
        pointer = config.model.pointer,
        device = device
    )
    start_epoch = 0
    if config.train.LOAD_EPOCH is not None:
        cpk = torch.load('checkpoints/%s/epoch_%04d.pth' % (config.name, config.train.LOAD_EPOCH))
        model.load_state_dict(cpk['model'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
        optimizer.load_state_dict(cpk['optimizer'])
        start_epoch = cpk['epoch']
        print('loaded', start_epoch, '!')
    else:
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    for epoch in range(start_epoch, config.train.epochs):
        print("epoch: %04d" % epoch)
        loss_avg, acc_avg = 0, 0
        total = len(train_loader)

        model = model.train()
        for i, (n, t, p) in enumerate(tqdm(train_loader)):
            n, t, p = n.to(device), t.to(device), p.to(device)
            optimizer.zero_grad()

            loss, ans = model(n, t, p)
            loss_avg += loss.item()
            acc_item = accuracy_score(t.cpu().numpy().flatten(), ans.cpu().numpy().flatten())
            acc_avg += acc_item
            loss.backward()

            if (i + 1) % 100 == 0:
                print('temp_loss: %f, temp_acc: %f' % (loss.item(), acc_item), flush=True)
#             if (i + 1) % 1000 == 0:
#                 break

            optimizer.step()

        print("avg_loss: %f, avg_acc: %f" % (loss_avg/total, acc_avg/total))

        if (epoch + 1) % config.train.eval_period == 0:
            with torch.no_grad():
                model = model.eval()
                acc = 0.
                loss_eval = 0.
                for i, (n, t, p) in enumerate(tqdm(test_loader)):
                    n, t, p = n.to(device), t.to(device), p.to(device)
                    loss, ans = model(n, t, p)
                    loss_eval += loss.item()
                    acc += accuracy_score(t.cpu().numpy().flatten(), ans.cpu().numpy().flatten())
                acc /= len(test_loader)
                loss_eval /= len(test_loader)
                print('avg acc:', acc, 'avg loss:', loss_eval)
        if (epoch + 1) % config.train.checkpoint_period == 0:
            os.system('mkdir -p checkpoints/' + config.name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoints/%s/epoch_%04d.pth' % (config.name, epoch))

if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        config = DotDict(yaml.safe_load(f))
    train(config)

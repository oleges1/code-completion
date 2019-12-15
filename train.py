from model import *
from data import *
import os
from tqdm import tqdm
import yaml
from utils import DotDict, adjust_learning_rate, accuracy
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import argparse

def train(config):
    writer = SummaryWriter('logs/' + config.name)

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
    
    ignored_index = data_train.vocab_sizeT - 1
    unk_index = data_train.vocab_sizeT - 2

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
        attn = config.model.attn,
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
        lr = config.train.lr * config.train.lr_decay ** max(epoch + 1 - config.train.epochs, 0.0)
        adjust_learning_rate(optimizer, lr)
        print("epoch: %04d" % epoch)
        loss_avg, acc_avg = 0, 0
        total = len(train_loader)

        model = model.train()
        for i, (n, t, p) in enumerate(tqdm(train_loader)):
            n, t, p = n.to(device), t.to(device), p.to(device)
            optimizer.zero_grad()

            loss, ans = model(n, t, p)
            loss_avg += loss.item()
            acc_item = accuracy(ans.cpu().numpy().flatten(), t.cpu().numpy().flatten(), ignored_index, unk_index)
            acc_avg += acc_item
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_value)
            loss.backward()

            if (i + 1) % 100 == 0:
                print('\ntemp_loss: %f, temp_acc: %f' % (loss.item(), acc_item), flush=True)
                writer.add_scalar('train/loss', loss.item(), epoch * total + i)
                writer.add_scalar('train/acc', acc_item, epoch * total + i)

#             if (i + 1) % 1000 == 0:
#                 break

            optimizer.step()

        print("\navg_loss: %f, avg_acc: %f" % (loss_avg/total, acc_avg/total))

        if (epoch + 1) % config.train.eval_period == 0:
            with torch.no_grad():
                model = model.eval()
                acc = 0.
                loss_eval = 0.
                for i, (n, t, p) in enumerate(tqdm(test_loader)):
                    n, t, p = n.to(device), t.to(device), p.to(device)
                    loss, ans = model(n, t, p)
                    loss_eval += loss.item()
                    acc += accuracy(ans.cpu().numpy().flatten(), t.cpu().numpy().flatten(), ignored_index, unk_index)
                acc /= len(test_loader)
                loss_eval /= len(test_loader)
                print('\navg acc:', acc, 'avg loss:', loss_eval)
                writer.add_scalar('val/loss', loss_eval, epoch)
                writer.add_scalar('val/acc', acc, epoch)
        if (epoch + 1) % config.train.checkpoint_period == 0:
            os.system('mkdir -p checkpoints/' + config.name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoints/%s/epoch_%04d.pth' % (config.name, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='configs/pointer_vocab_10k.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = DotDict(yaml.safe_load(f))
    train(config)

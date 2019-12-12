# code-completion
Pytorch version of code completion with neural attention and pointer networks


## TO DO LIST:
- [ ] refactor preprocessing code
- [x] fix data yelding, len dataset
- [x] add tensorboard
- [x] fix accuracy calculation
- [x] fix loss calculation (strange problems woth NLLLoss)
- [x] add simplier models
- [ ] add python to AST code
- [ ] config for preprocessing

## Requirments list:

- python3 >= 3.6
- torch >= 1.2, or tensorboadX for earlier versions
- pyyaml


## Instruction:

- run `python3 preprocess.py` for preprocessing
- run `CUDA_VISIBLE_DEVICES=id python3 train.py` for training with specified config, list of available configscan be found at configs folder


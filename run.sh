rm -r logs/pointer_vocab_50k
# python3 train.py --config=configs/pointer_vocab_10k.yml
python train.py --config=configs/pointer_vocab_50k.yml
python train.py --config=configs/attn_lstm_vocab_1k.yml
python train.py --config=configs/attn_lstm_vocab_50k.yml
#python3 train.py --config=configs/simple_lstm_vocab_1k.yml
python train.py --config=configs/simple_lstm_vocab_50k.yml
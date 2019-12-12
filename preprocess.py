import os
# import argparse

if __name__ == '__main__':
    os.system('mkdir -p pickle_data')
    os.system('python preprocess_utils/freq_dict.py')
    os.system('python preprocess_utils/get_non_terminal.py')
    os.system('python preprocess_utils/get_terminal_dict.py')
    os.system('python preprocess_utils/get_terminal_whole.py')

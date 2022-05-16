import argparse
import csv
from tqdm import tqdm
import torch
import torchaudio


def extract_length(input_file):
    wav, _ = torchaudio.load(input_file)
    return wav.size(-1)


def load_wsj_flist(path):
    print(f'Loading file list from {path}')
    with open(path, 'r') as fp:
        data_list = []
        for line in tqdm(fp):
            line = line.strip('\n')
            line = line.replace('/work/harry87122/',
                                '/work/harry87122/dataset/')
            line = line.replace('wv1', 'wav')
            data_list.append(line)
        return data_list


def get_all_length(data):
    print('Extracting lengths of audio files')
    len_list = []
    for d in tqdm(data):
        len_list.append(extract_length(d))
    return len_list


def save_to_csv(path, data_list, len_list):
    print(f'Saving results to {path}')
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['', 'file_path', 'length', 'label'])
        for i, (f, l) in tqdm(enumerate(zip(data_list, len_list))):
            writer.writerow([i, f, l, ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsj', type=str, help='Path to .flist of WSJ')
    parser.add_argument('--out', type=str, help='Path to save .csv')
    args = parser.parse_args()

    files = load_wsj_flist(args.wsj)
    lens = get_all_length(files)
    files, lens = zip(
        *sorted(zip(files, lens), reverse=True, key=lambda x: x[1]))
    save_to_csv(args.out, files, lens)

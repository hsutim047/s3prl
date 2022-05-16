import argparse
import csv
import os
from os.path import join, getsize
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio


def extract_length(input_file):
    wav, sample_rate = torchaudio.load(input_file)
    #print('sample rate: ', sample_rate) # 16000
    return wav.size(-1)


def load_transcript(path):
    with open(path, 'r') as fp:
        idx_list = []
        for line in fp:
            line = line.strip('\n')
            idx = line.split(' ')[0]
            if int(idx[8:11]) >= 2 and int(idx[8:11]) <= 723:
                idx_list.append(idx)
        return idx_list


def find_all_files(corpus_dir, idx_list):
    idx_list = set(idx_list)
    data_list = list(Path(join(corpus_dir, 'wav/train')).rglob("*.wav"))
    data_list = [str(f) for f in data_list
                 if str(f).split('/')[-1][:-4] in idx_list]
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
    parser.add_argument('--aishell', type=str, help='Directory of AISHELL/')
    parser.add_argument('--out', type=str, help='Path to save .csv')
    args = parser.parse_args()

    idxs = load_transcript(
        join(args.aishell, 'transcript', 'aishell_transcript_v0.8.txt'))
    files = find_all_files(args.aishell, idxs)
    lens = get_all_length(files)

    files, lens = zip(
        *sorted(zip(files, lens), reverse=True, key=lambda x: x[1]))
    save_to_csv(args.out, files, lens)

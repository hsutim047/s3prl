import argparse
import csv
import os
from pathlib import Path
import torch
import torchaudio
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=521, help='random seed')
    parser.add_argument('--meta_data_path', type=str, default='../data/aishell_csv/aishell.csv', help='directory of metadata of full dataset')
    parser.add_argument('--out_path', type=str, default='../data/aishell_csv/aishell-10h.csv', help='directory of metadata of extracted subset')
    parser.add_argument('--hour', type=int, default=10, help='length (unit: hour) of generated subset')
    args = parser.parse_args()
    
    random.seed(args.seed)

    meta_data_info = []
    total_data_length = 0
    with open(args.meta_data_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        header = next(rows, None) # extract the header
        for row in rows:
            # row[0] is file id, row[1] is file path, row[2] is record length
            row[0] = int(row[0])
            row[2] = int(row[2])
            meta_data_info.append(row)
            total_data_length += row[2]
        
    if 'aishell' in args.meta_data_path.lower():
        args.sample_rate = 16000
    else:
        raise Exception('sataset not implemented yet')

    print('original data total length (hrs): {}'.format(total_data_length / 3600 / args.sample_rate))
    print('ideal extracted subset length (hrs): {}'.format(args.hour))
    
    args.subsetSampleLen = args.hour * 3600 * args.sample_rate # total number of data points of subset, assume to be 576000000
    subset_meta_data_info = []
    subset_data_length = 0
    while subset_data_length < args.subsetSampleLen:
        rd = random.randint(0, len(meta_data_info)-1)
        subset_meta_data_info.append(meta_data_info[rd])
        subset_data_length += meta_data_info[rd][2]
        del meta_data_info[rd]

    print('final extracted subset length (hrs): {}'.format(subset_data_length / 3600 / args.sample_rate))
    subset_meta_data_info.sort(key=lambda s: s[2], reverse=True) # sort by data (file) length

    for i in range(len(subset_meta_data_info)):
        subset_meta_data_info[i][0] = i # reset the file id    

    print('number of files in the subset: {}'.format(subset_meta_data_info[-1][0]+1))
        
    with open(args.out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(subset_meta_data_info)
    
    print('subset meta-data written to {}'.format(args.out_path))
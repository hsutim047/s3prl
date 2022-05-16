import argparse
import csv
import os
from pathlib import Path
import torch
import torchaudio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_data_path', type=str, default='../data/aishell_csv/aishell.csv', help='directory of metadata of full dataset')
    parser.add_argument('--out_path', type=str, default='../data/aishell_csv/aishell-10h.csv', help='directory of metadata of 10h subset')
    parser.add_argument('--hour', type=int, default=10, help='length (unit: hour) of generated subset')
    args = parser.parse_args()

    meta_data_info = []
    with open(args.meta_data_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            meta_data_info.append(row)
        
        if 'aishell' in args.meta_data_path.lower():
            sample_rate = 16000
        else:
            raise Exception('sataset not implemented yet')
        
        args.subsetSampleLen = args.hour * 3600 * sample_rate
        subset_meta_data_info = []


        print(rows)



